# ---------------------------------------------------------
# chunk_sections.py
#
# Turns preprocessed documentation sections into retrieval-ready
# chunks for embedding (and later BM25).
#
# Input:  processed/sections.jsonl  (one record per heading section,
#                                    produced by preprocess_docs.py)
# Output: processed/chunks.jsonl    (one record per chunk)
#
# Strategy (structure-aware, measured in tokens):
#   1. Clean sections
#      - drop HTML-only navigation cards that contain no link
#        (cards with links are reference lists and are kept)
#      - drop raw OpenAPI JSON (safety net only: preprocess_docs.py
#        already converts specs into readable text)
#      - Glossary of Terms: drop the "A B C ..." letter index and
#        fold "See X" entries into X as an alias line
#   2. Merge small neighbouring sections that share a parent heading
#      until a chunk reaches ~TARGET_TOKENS. Never across documents.
#      Glossary terms are never merged: one term = one chunk.
#      Tail merge: a leftover chunk under MIN_CHUNK_TOKENS (e.g. a
#      "Next steps: see X" pointer) is appended to the previous chunk
#      of the same document, if the result stays within MAX_TOKENS.
#   3. Split sections larger than MAX_TOKENS at block boundaries
#      (paragraph / list / table / code), so no chunk ends abruptly:
#      - a block that fits in one chunk is never cut
#      - a lead-in line ("Request body fields:") stays with the
#        block it introduces
#      - an oversized block is split structurally: lists by item
#        (parent items repeated), tables by row (header row
#        repeated), code by line (fences re-added), prose by
#        sentence with ~OVERLAP_TOKENS of overlap
#   4. Start every chunk with its breadcrumb and heading path
#      ("Product > Page > Section > Sub") so even short chunks carry
#      their context when embedded.
#      API reference chunks also name the API and the operation,
#      e.g. "In-Memory DB API > Create ReplicaSet (POST /replicasets)".
#      API category pages without a spec of their own (e.g. "Restore")
#      borrow the API name from the endpoint pages below them.
#
# Tokens are counted with tiktoken's cl100k_base encoding, the tokenizer
# used by OpenAI's text-embedding-3 models, so counts match the API.
# tiktoken is only used as a ruler; chunks are stored as plain text.
#
# Usage:
#   python scripts/chunk_sections.py
#   python scripts/chunk_sections.py --input sections.jsonl --output chunks.jsonl
# ---------------------------------------------------------

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import tiktoken

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROCESSED_DIR = Path.home() / "Projects" / "ionos-docs" / "processed"

DEFAULT_INPUT = PROCESSED_DIR / "sections.jsonl"
DEFAULT_OUTPUT = PROCESSED_DIR / "chunks.jsonl"


# ---------------------------------------------------------
# Chunk sizes (tokens)
# ---------------------------------------------------------

TARGET_TOKENS = 500  # stop merging small sections once a chunk reaches this
MAX_TOKENS = 800  # hard upper bound; larger sections are split
OVERLAP_TOKENS = 100  # overlap carried between split prose pieces
MIN_FIRST_PIECE_TOKENS = 100  # smallest leftover space worth starting a split block in
LEAD_IN_MAX_TOKENS = 50  # a short block ending in ":" that introduces the next block
MIN_CHUNK_TOKENS = 50  # smaller leftover chunks are appended to the previous chunk

GLOSSARY_URL_SUFFIX = "/glossary-of-terms"


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


# ---------------------------------------------------------
# Blocks
#
# preprocess_docs.py joins the blocks of a section with a blank
# line, so we can recover them by splitting on blank lines, as long
# as we don't split inside fenced code blocks.
# ---------------------------------------------------------

FENCE = "```"

LIST_ITEM_RE = re.compile(r"^\s{0,3}(?:[-*+]|\d+[.)])\s")

LIST_MARKER_RE = re.compile(r"(?:[-*+]|\d+[.)])\s")

SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# Lines written by preprocess_docs.py for converted OpenAPI specs.
API_TITLE_RE = re.compile(r"^API: (.+)$", re.MULTILINE)
API_OPERATION_RE = re.compile(r"^(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS) /\S*$", re.MULTILINE)


def fence_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.lstrip().startswith(FENCE))


def split_blocks(content: str) -> list[str]:
    blocks = []
    current: list[str] = []
    in_code = False

    for part in content.split("\n\n"):
        current.append(part)

        if fence_count(part) % 2 == 1:
            in_code = not in_code

        if not in_code:
            blocks.append("\n\n".join(current))
            current = []

    if current:
        blocks.append("\n\n".join(current))

    return [block for block in blocks if block.strip()]


def classify_block(block: str, section_block_types: list[str]) -> str:
    stripped = block.strip()
    first_line = stripped.splitlines()[0]

    if stripped.startswith(FENCE):
        return "code"

    if first_line.startswith("|") and len(stripped.splitlines()) > 1:
        return "table"

    if LIST_ITEM_RE.match(first_line):
        return "list"

    # HTML blocks are rendered as plain text by preprocess_docs.py,
    # so they look like paragraphs. Keep the original label when the
    # section had HTML but no Markdown paragraphs.
    if "paragraph" not in section_block_types and "html" in section_block_types:
        return "html"

    return "paragraph"


def is_openapi_spec(block: str) -> bool:
    stripped = block.lstrip()
    return stripped.startswith(FENCE) and '"openapi"' in stripped[:200]


def api_context(section: dict[str, Any]) -> tuple[str, str | None] | None:
    """
    For a section converted from an OpenAPI spec: the API name and,
    on endpoint pages, the operation (e.g. "POST /replicasets").
    Used in the chunk header, because page titles such as "Models"
    or "Create Cluster" do not say which product they belong to.
    """

    if "api_spec" not in section["block_types"]:
        return None

    title = API_TITLE_RE.search(section["content"])

    if not title:
        return None

    operation = API_OPERATION_RE.search(section["content"])

    return title.group(1), operation.group(0) if operation else None


# ---------------------------------------------------------
# Section cleanup
# ---------------------------------------------------------

SEE_ALSO_RE = re.compile(r"^See (.+?)\s*\.?$")


def prepare_section(section: dict[str, Any], stats: Counter) -> dict[str, Any] | None:
    """
    Convert a sections.jsonl record into the internal form used by the
    chunker, or return None if the whole section is noise.
    """

    # HTML-only sections are GitBook cards. A card without any link is
    # a navigation label ("FAQ Get answers ... FAQ") and is dropped.
    # Cards with links are reference lists (e.g. the catalogue of all
    # API specification files) and are kept.
    if section["block_types"] == ["html"] and "http" not in section["content"]:
        stats["dropped_html_nav_sections"] += 1
        return None

    blocks = []

    for text in split_blocks(section["content"]):

        if is_openapi_spec(text):
            stats["dropped_openapi_blocks"] += 1
            continue

        blocks.append((classify_block(text, section["block_types"]), text.strip()))

    if not blocks:
        stats["dropped_empty_sections"] += 1
        return None

    return {
        "section_ids": [section["section_id"]],
        "document_title": section["document_title"],
        "source_url": section["source_url"],
        "heading": section["heading"],
        "heading_path": section["heading_path"],
        "breadcrumb": section.get("breadcrumb", []),
        "blocks": blocks,
        "aliases": [],
        "api_context": api_context(section),
    }


def find_glossary_term(term: str, by_heading: dict[str, dict]) -> dict | None:
    """
    "See Application Load Balancer" points at the heading
    "Application Load Balancer (ALB)", so also match with an
    acronym suffix.
    """

    if term in by_heading:
        return by_heading[term]

    for heading, section in by_heading.items():
        if heading.startswith(f"{term} ("):
            return section

    return None


def clean_glossary(sections: list[dict], stats: Counter) -> list[dict]:
    by_heading = {section["heading"]: section for section in sections}

    kept = []

    for section in sections:

        # The page-level section is only the "A B C ..." letter index.
        if len(section["heading_path"]) == 1:
            stats["dropped_glossary_index"] += 1
            continue

        text = "\n\n".join(block for _, block in section["blocks"])
        match = SEE_ALSO_RE.match(text)
        target = find_glossary_term(match.group(1), by_heading) if match else None

        if target is not None and target is not section:
            target["aliases"].append(section["heading"])
            target["section_ids"].append(section["section_ids"][0])
            stats["folded_glossary_aliases"] += 1
            continue

        kept.append(section)

    return kept


def prepare_document(sections: list[dict[str, Any]], stats: Counter) -> list[dict[str, Any]]:
    """
    Clean all sections of one document, in their original order.
    """

    prepared = [prepare_section(section, stats) for section in sections]
    prepared = [section for section in prepared if section is not None]

    if is_glossary(sections):
        prepared = clean_glossary(prepared, stats)

    return prepared


def is_glossary(sections: list[dict[str, Any]]) -> bool:
    return bool(sections) and sections[0]["source_url"].endswith(GLOSSARY_URL_SUFFIX)


# ---------------------------------------------------------
# Chunk text
# ---------------------------------------------------------


def common_prefix(paths: list[list[str]]) -> list[str]:
    prefix = []

    for parts in zip(*paths):
        if len(set(parts)) != 1:
            break
        prefix.append(parts[0])

    return prefix


def section_body(section: dict[str, Any], chunk_path: list[str]) -> str:
    """
    Body of one section inside a chunk. When several sections are
    merged, each keeps its own sub-heading (e.g. an FAQ question).
    """

    body = "\n\n".join(text for _, text in section["blocks"])

    if section["aliases"]:
        body += "\n\nAlso known as: " + ", ".join(section["aliases"]) + "."

    sub_path = section["heading_path"][len(chunk_path) :]

    if sub_path:
        body = f"## {' > '.join(sub_path)}\n\n{body}"

    return body


def with_breadcrumb(sections: list[dict[str, Any]], header: str) -> str:
    """
    Prefix the titles of the page's parent pages, e.g.
    "Error Codes > API errors" -> "AI Model Hub > Error Codes > API errors".
    All sections of a chunk come from one page, so they share it.
    """

    crumbs = sections[0]["breadcrumb"]

    return " > ".join(crumbs + [header]) if crumbs else header


def chunk_header(sections: list[dict[str, Any]], chunk_path: list[str]) -> str:
    """
    First line of every chunk: the breadcrumb and heading path.
    Chunks of API reference sections also name the API and the
    operation, e.g. "In-Memory DB v1 > Replica Set > In-Memory DB API
    (version 1.0.0) > Create ReplicaSet (POST /replicasets)".
    """

    header = " > ".join(chunk_path) if chunk_path else sections[0]["document_title"]

    contexts = {section["api_context"] for section in sections}

    if len(contexts) == 1 and None not in contexts:
        api_title, operation = contexts.pop()

        if operation:
            header += f" ({operation})"

        header = f"{api_title} > {header}"

    return with_breadcrumb(sections, header)


def make_chunk(
    sections: list[dict[str, Any]],
    chunk_path: list[str],
    header: str,
    body: str,
    block_types: set[str],
    split: bool = False,
) -> dict[str, Any]:
    """
    A chunk record. Keys starting with "_" are only used while
    chunking (e.g. by the tail merge) and are not written out.
    """

    text = f"{header}\n\n{body}"

    return {
        "document_title": sections[0]["document_title"],
        "source_url": sections[0]["source_url"],
        "breadcrumb": sections[0]["breadcrumb"],
        "section_ids": [sid for section in sections for sid in section["section_ids"]],
        "heading_path": chunk_path,
        "block_types": sorted(block_types),
        "token_count": count_tokens(text),
        "text": text,
        "_sections": sections,
        "_body": body,
        "_split": split,
    }


def merged_chunk(group: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_path = common_prefix([section["heading_path"] for section in group])

    body = "\n\n".join(section_body(section, chunk_path) for section in group)

    block_types = {block_type for section in group for block_type, _ in section["blocks"]}

    return make_chunk(
        group,
        chunk_path,
        chunk_header(group, chunk_path),
        body,
        block_types,
    )


def merged_token_count(group: list[dict[str, Any]]) -> int:
    return merged_chunk(group)["token_count"]


def glossary_chunk(section: dict[str, Any]) -> dict[str, Any]:
    # The letter level ("A", "B", ...) adds nothing to the header.
    header = with_breadcrumb([section], f"{section['document_title']} > {section['heading']}")

    return make_chunk(
        [section],
        section["heading_path"],
        header,
        section_body(section, section["heading_path"]),
        {block_type for block_type, _ in section["blocks"]},
    )


# ---------------------------------------------------------
# Splitting oversized sections
# ---------------------------------------------------------


def hard_split(text: str, budget: int) -> list[str]:
    """
    Last resort for a single unit (line, sentence, list item) that is
    larger than the budget on its own, e.g. a minified JSON line.
    """

    tokens = encoding.encode(text)

    if len(tokens) <= budget:
        return [text]

    return [encoding.decode(tokens[i : i + budget]) for i in range(0, len(tokens), budget)]


def pack(
    units: list[str],
    budget: int,
    sep: str,
    prefix: str = "",
    suffix: str = "",
    first_budget: int | None = None,
) -> list[str]:
    """
    Greedily join units with sep into pieces of at most budget tokens,
    each wrapped in prefix/suffix (used for repeated table headers and
    code fences). The first piece may use a smaller first_budget so it
    fits after the blocks already in the chunk.
    """

    unit_budget = budget - count_tokens(prefix + suffix)

    pieces = []
    current: list[str] = []

    for unit in units:
        for part in hard_split(unit, unit_budget):

            candidate = current + [part]
            limit = first_budget if first_budget is not None and not pieces else budget

            if current and count_tokens(prefix + sep.join(candidate) + suffix) > limit:
                pieces.append(prefix + sep.join(current) + suffix)
                current = [part]
            else:
                current = candidate

    if current:
        pieces.append(prefix + sep.join(current) + suffix)

    return pieces


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def list_items(block: str) -> list[str]:
    """
    Every item of a list, at any nesting level, each with its
    continuation lines.
    """

    items: list[str] = []

    for line in block.splitlines():
        if LIST_MARKER_RE.match(line.lstrip()) or not items:
            items.append(line)
        else:
            items[-1] += "\n" + line

    return items


def split_list(block: str, budget: int, first_budget: int) -> list[str]:
    """
    Split a list between items. When a piece starts inside a nested
    list, its parent items are repeated at the top (like a table
    header row), so nested items never lose what they belong to.
    """

    pieces: list[str] = []
    current: list[str] = []
    parents: list[str] = []  # items enclosing the current one

    for item in list_items(block):

        parents = [parent for parent in parents if indent_of(parent) < indent_of(item)]
        parent_tokens = count_tokens("\n".join(parents) + "\n") if parents else 0

        for part in hard_split(item, budget - parent_tokens):

            limit = budget if pieces else first_budget

            if current and count_tokens("\n".join(current + [part])) > limit:

                # A parent left at the end would be cut off from its
                # nested items; it is repeated on the next piece anyway.
                while current and current[-1] in parents:
                    current.pop()

                if current:
                    pieces.append("\n".join(current))

                current = parents + [part]
            else:
                current.append(part)

        parents.append(item.splitlines()[0])

    if current:
        pieces.append("\n".join(current))

    return pieces


def split_block(block_type: str, block: str, budget: int, first_budget: int) -> list[str]:
    """
    Pieces of a block: the first fits first_budget (the space left in
    the current chunk), the others fit budget. A block that fits the
    space left is returned whole.
    """

    if count_tokens(block) <= first_budget:
        return [block]

    lines = block.splitlines()

    if block_type == "table":
        header = "\n".join(lines[:2]) + "\n"
        return pack(lines[2:], budget, "\n", prefix=header, first_budget=first_budget)

    if block_type == "code":
        opening = lines[0]
        has_closing = len(lines) > 1 and lines[-1].strip() == FENCE
        inner = lines[1:-1] if has_closing else lines[1:]
        return pack(
            inner, budget, "\n", prefix=opening + "\n", suffix="\n" + FENCE, first_budget=first_budget
        )

    if block_type == "list":
        return split_list(block, budget, first_budget)

    return pack(SENTENCE_END_RE.split(block), budget, " ", first_budget=first_budget)


def overlap_tail(text: str) -> str:
    """
    The last sentences of a paragraph, up to OVERLAP_TOKENS.
    """

    tail: list[str] = []

    for sentence in reversed(SENTENCE_END_RE.split(text)):
        if count_tokens(" ".join([sentence] + tail)) > OVERLAP_TOKENS:
            break
        tail.insert(0, sentence)

    return " ".join(tail)


def is_lead_in(unit: tuple[str, str]) -> bool:
    """
    A short line such as "Request body fields:" or "The table below
    lists the limits:" that only introduces the block after it.
    """

    block_type, text = unit

    return (
        block_type in {"paragraph", "html"}
        and text.rstrip().endswith(":")
        and count_tokens(text) <= LEAD_IN_MAX_TOKENS
    )


def split_section(section: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Split one oversized section into several chunks, only at
    structure boundaries, so that no chunk ends abruptly:
      - blocks stay whole when they fit in a chunk
      - a lead-in line always stays with the block it introduces
      - oversized blocks are split structurally (see split_block)
      - consecutive prose pieces overlap by ~OVERLAP_TOKENS
    """

    chunk_path = section["heading_path"]
    header = chunk_header([section], chunk_path)
    budget = MAX_TOKENS - count_tokens(header + "\n\n")

    blocks = list(section["blocks"])

    if section["aliases"]:
        blocks.append(("paragraph", "Also known as: " + ", ".join(section["aliases"]) + "."))

    def join(units: list[tuple[str, str]]) -> str:
        return "\n\n".join(text for _, text in units)

    def fits(units: list[tuple[str, str]]) -> bool:
        return count_tokens(join(units)) <= budget

    def space_after(units: list[tuple[str, str]]) -> int:
        return budget - count_tokens(join(units) + "\n\n") if units else budget

    def trailing_lead_ins(units: list[tuple[str, str]]) -> list[tuple[str, str]]:
        lead_ins: list[tuple[str, str]] = []
        for unit in reversed(units):
            if not is_lead_in(unit):
                break
            lead_ins.insert(0, unit)
        return lead_ins

    pieces: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []

    def close_chunk(next_unit: tuple[str, str], next_will_be_split: bool = False):
        """
        End the current chunk before next_unit. Lead-ins at its end
        move to the new chunk; prose followed by prose gets overlap.
        Both are only added if next_unit still fits, unless it is
        about to be split to the space that is left anyway.
        """
        nonlocal current

        carry = trailing_lead_ins(current)

        if carry and not (next_will_be_split or fits(carry + [next_unit])):
            carry = []

        closed = current[: len(current) - len(carry)]

        if not closed:
            return

        pieces.append(closed)
        current = carry

        last_type, last_text = closed[-1]

        if not carry and last_type == "paragraph" and next_unit[0] == "paragraph":
            overlap = ("paragraph", overlap_tail(last_text))

            if overlap[1] and (next_will_be_split or fits([overlap, next_unit])):
                current = [overlap]

    for block_type, block in blocks:

        unit = (block_type, block)

        # A block that fits in one chunk (together with its lead-in)
        # moves to a new chunk instead of being split. An oversized
        # block starts in the space left here, if that is worth it.
        if current and not fits(current + [unit]):
            if fits(trailing_lead_ins(current) + [unit]):
                close_chunk(unit)
            elif space_after(current) < MIN_FIRST_PIECE_TOKENS:
                close_chunk(unit, next_will_be_split=True)

        # Pieces of a split block leave room for the lead-in they may
        # carry along and, for prose, for the overlap.
        piece_budget = space_after(trailing_lead_ins(current))

        if block_type == "paragraph":
            piece_budget -= OVERLAP_TOKENS

        for piece in split_block(block_type, block, piece_budget, space_after(current)):

            unit = (block_type, piece)

            if current and not fits(current + [unit]):
                close_chunk(unit)

            current.append(unit)

    if current:
        pieces.append(current)

    return [
        make_chunk(
            [section],
            chunk_path,
            header,
            join(piece),
            {block_type for block_type, _ in piece},
            split=True,
        )
        for piece in pieces
    ]


# ---------------------------------------------------------
# Chunk one document
# ---------------------------------------------------------


def can_merge(group: list[dict[str, Any]], section: dict[str, Any]) -> bool:
    """
    A section may join the group only if it stays under the parent
    heading of the group's first section (never below the document).
    """

    first_path = group[0]["heading_path"]
    parent = first_path[:-1] or first_path[:1]

    return section["heading_path"][: len(parent)] == parent


def absorb_tail(previous: dict[str, Any], group: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Append a tiny leftover group of sections (e.g. "Explore further:
    see Text Embeddings") to the previous chunk of the same document.
    Unlike normal merging, the leftover does not need the same parent
    heading: a page-level "Next steps" may follow a chunk about a
    sub-section. The combined heading path is then the shared part
    (the page title) and sub-headings move into the body.
    Returns the combined chunk, or None if it would exceed MAX_TOKENS.
    """

    sections = previous["_sections"] + group

    if not common_prefix([section["heading_path"] for section in sections]):
        return None

    if not previous["_split"]:
        combined = merged_chunk(sections)
    else:
        # The previous chunk is the last piece of a split section, so
        # its body is re-used as is. If the combined heading path is
        # shorter, the piece's own heading moves into the body.
        chunk_path = common_prefix([section["heading_path"] for section in sections])
        body = previous["_body"]

        own_path = previous["heading_path"][len(chunk_path) :]
        if own_path:
            body = f"## {' > '.join(own_path)}\n\n{body}"

        body += "\n\n" + "\n\n".join(section_body(section, chunk_path) for section in group)

        block_types = set(previous["block_types"]) | {
            block_type for section in group for block_type, _ in section["blocks"]
        }

        combined = make_chunk(
            sections,
            chunk_path,
            chunk_header(sections, chunk_path),
            body,
            block_types,
            split=True,
        )

    return combined if combined["token_count"] <= MAX_TOKENS else None


def chunk_document(
    sections: list[dict[str, Any]],
    glossary: bool,
    stats: Counter | None = None,
) -> list[dict[str, Any]]:
    if glossary:
        return [glossary_chunk(section) for section in sections]

    chunks: list[dict[str, Any]] = []
    group: list[dict[str, Any]] = []

    def flush_group():
        nonlocal group
        if group:
            chunk = merged_chunk(group)

            # Tail merge: a tiny leftover joins the previous chunk
            # instead of becoming a chunk on its own.
            combined = None
            if chunk["token_count"] < MIN_CHUNK_TOKENS and chunks:
                combined = absorb_tail(chunks[-1], group)

            if combined is not None:
                chunks[-1] = combined
                if stats is not None:
                    stats["absorbed_tail_chunks"] += 1
            else:
                chunks.append(chunk)
        group = []

    for section in sections:

        if merged_token_count([section]) > MAX_TOKENS:
            flush_group()
            chunks.extend(split_section(section))
            continue

        if (
            group
            and can_merge(group, section)
            and merged_token_count(group) < TARGET_TOKENS
            and merged_token_count(group + [section]) <= MAX_TOKENS
        ):
            group.append(section)
            continue

        flush_group()
        group = [section]

    flush_group()

    return chunks


# ---------------------------------------------------------
# Load / write
# ---------------------------------------------------------


def document_id(section_id: str) -> str:
    return section_id.rsplit(":section:", 1)[0]


def inherited_api_titles(documents: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    """
    API reference category pages such as "Restore" or "Routes" have no
    spec of their own, so their chunks would not say which product they
    belong to. They borrow the API name from the endpoint pages below
    them in the docs, e.g. .../in-memory-db/restore borrows "In-Memory
    DB API" from .../in-memory-db/restore/create-restore.

    Returns {source_url: api_title} for those pages.
    """

    api_titles = {}

    for sections in documents.values():
        for section in sections:
            context = api_context(section)
            if context:
                api_titles[section["source_url"]] = context[0]

    inherited = {}

    for sections in documents.values():
        url = sections[0]["source_url"]

        if url in api_titles:
            continue

        titles = {title for page, title in api_titles.items() if page.startswith(url + "/")}

        # Only when all pages below agree on one API.
        if len(titles) == 1:
            inherited[url] = titles.pop()

    return inherited


def load_documents(path: Path) -> dict[str, list[dict[str, Any]]]:
    """
    Group sections by document, keeping file order (which is the
    section order within each document).
    """

    documents: dict[str, list[dict[str, Any]]] = {}

    with path.open(encoding="utf-8") as file:
        for line in file:
            section = json.loads(line)
            documents.setdefault(document_id(section["section_id"]), []).append(section)

    return documents


def main():

    parser = argparse.ArgumentParser(description="Chunk sections.jsonl into chunks.jsonl")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Sections file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    documents = load_documents(args.input)
    inherited = inherited_api_titles(documents)

    stats: Counter = Counter()
    total_chunks = 0

    with args.output.open("w", encoding="utf-8") as output:

        for doc_id, sections in documents.items():

            stats["sections_in"] += len(sections)

            prepared = prepare_document(sections, stats)

            api_title = inherited.get(sections[0]["source_url"])
            if api_title:
                stats["api_name_inherited_documents"] += 1
                for section in prepared:
                    section["api_context"] = (api_title, None)

            chunks = chunk_document(prepared, is_glossary(sections), stats)

            for index, chunk in enumerate(chunks):
                record = {"chunk_id": f"{doc_id}:chunk:{index}", "chunk_index": index}
                record.update({key: value for key, value in chunk.items() if not key.startswith("_")})
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_chunks += len(chunks)

    print("=" * 50)
    print("Chunking complete")
    print("=" * 50)
    print(f"Documents: {len(documents)}")
    for key, value in sorted(stats.items()):
        print(f"{key + ':':<32}{value}")
    print(f"Chunks:    {total_chunks}")
    print(f"Output:    {args.output}")


if __name__ == "__main__":
    main()
