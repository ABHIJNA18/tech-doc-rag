# ---------------------------------------------------------
# inspect_chunks.py
#
# Validation and inspection for chunks.jsonl (output of
# chunk_sections.py). Run it after chunking and before embedding.
#
# It prints:
#   - validation checks: required fields, unique chunk IDs, valid
#     source URLs, stored token_count matches a recount, no chunk
#     above MAX_TOKENS, no unbalanced code fences, no chunk that
#     ends abruptly (on a dangling lead-in line or a bare heading)
#   - coverage checks against sections.jsonl: every section that
#     survived cleanup appears in a chunk, and every word of its
#     content appears in its chunks (nothing silently lost)
#   - statistics: token histogram, chunks per block type,
#     smallest / largest chunks, largest documents
#   - random sample chunks (or all chunks of one document) to read
#
# Usage:
#   python scripts/inspect_chunks.py
#   python scripts/inspect_chunks.py --samples 10 --seed 7
#   python scripts/inspect_chunks.py --doc glossary-of-terms
#   python scripts/inspect_chunks.py --chunks path/to/chunks.jsonl --sections path/to/sections.jsonl
# ---------------------------------------------------------

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from chunk_sections import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    MAX_TOKENS,
    count_tokens,
    fence_count,
    load_documents,
    prepare_document,
)

REQUIRED_FIELDS = [
    "chunk_id",
    "chunk_index",
    "document_title",
    "source_url",
    "breadcrumb",
    "section_ids",
    "heading_path",
    "block_types",
    "token_count",
    "text",
]

URL_PREFIX = "https://docs.ionos.com/cloud/"

TOKEN_BUCKETS = [50, 100, 200, 300, 400, 500, 600, 700, 800]

WORD_RE = re.compile(r"\w+")

MAX_EXAMPLES = 5


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------


def validate(chunks: list[dict[str, Any]]) -> dict[str, list[str]]:
    problems: dict[str, list[str]] = {}

    def report(check: str, message: str):
        problems.setdefault(check, []).append(message)

    seen_ids = set()

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "<missing id>")

        missing = [field for field in REQUIRED_FIELDS if field not in chunk]
        if missing:
            report("missing fields", f"{chunk_id}: {missing}")
            continue

        if chunk_id in seen_ids:
            report("duplicate chunk_id", chunk_id)
        seen_ids.add(chunk_id)

        if not chunk["source_url"].startswith(URL_PREFIX):
            report("invalid source_url", f"{chunk_id}: {chunk['source_url']}")

        if not chunk["text"].strip():
            report("empty text", chunk_id)

        if count_tokens(chunk["text"]) != chunk["token_count"]:
            report("token_count mismatch", chunk_id)

        if chunk["token_count"] > MAX_TOKENS:
            report(f"over {MAX_TOKENS} tokens", f"{chunk_id}: {chunk['token_count']}")

        if fence_count(chunk["text"]) % 2 != 0:
            report("unbalanced code fence", chunk_id)

        # A chunk must not end abruptly: not on a line that only
        # introduces what follows, and not on a heading without content.
        last_line = chunk["text"].rstrip().splitlines()[-1].strip()

        if last_line.endswith(":") and not last_line.startswith(("-", "|")):
            report("ends with a dangling lead-in", f"{chunk_id}: {last_line[:80]!r}")

        if last_line.startswith("## "):
            report("ends with a heading", f"{chunk_id}: {last_line[:80]!r}")

    return problems


def check_coverage(chunks: list[dict[str, Any]], sections_path: Path) -> dict[str, list[str]]:
    """
    Re-run the same cleanup as the chunker, then check that every
    kept section, and every word in it, made it into a chunk.
    """

    problems: dict[str, list[str]] = {}

    chunk_words: dict[str, Counter] = {}

    for chunk in chunks:
        words = Counter(WORD_RE.findall(chunk["text"]))
        for section_id in chunk["section_ids"]:
            chunk_words.setdefault(section_id, Counter()).update(words)

    for sections in load_documents(sections_path).values():

        for section in prepare_document(sections, Counter()):

            section_id = section["section_ids"][0]

            if section_id not in chunk_words:
                problems.setdefault("section not in any chunk", []).append(section_id)
                continue

            text = "\n\n".join(block for _, block in section["blocks"])
            missing = Counter(WORD_RE.findall(text)) - chunk_words[section_id]

            if missing:
                problems.setdefault("words missing from chunks", []).append(
                    f"{section_id}: {list(missing)[:8]}"
                )

    return problems


def print_problems(title: str, problems: dict[str, list[str]]):
    print(f"\n{title}")
    print("-" * 50)

    if not problems:
        print("OK - no problems found")
        return

    for check, messages in problems.items():
        print(f"FAIL {check}: {len(messages)}")
        for message in messages[:MAX_EXAMPLES]:
            print(f"     {message}")


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------


def print_stats(chunks: list[dict[str, Any]]):
    tokens = sorted(chunk["token_count"] for chunk in chunks)

    print("\nStatistics")
    print("-" * 50)
    print(f"Chunks:    {len(chunks)}")
    print(f"Documents: {len({chunk['source_url'] for chunk in chunks})}")
    print(f"Tokens:    total {sum(tokens)}, min {tokens[0]}, median {statistics.median(tokens):.0f}, "
          f"p90 {tokens[int(0.9 * len(tokens))]}, max {tokens[-1]}")

    print("\nToken histogram")
    lower = 0
    for upper in TOKEN_BUCKETS:
        count = sum(lower < t <= upper for t in tokens)
        print(f"  {lower + 1:>4}-{upper:<4} {count:>4}  {'#' * (count // 2)}")
        lower = upper
    print(f"  {'>' + str(lower):>9} {sum(t > lower for t in tokens):>4}")

    print("\nChunks containing each block type")
    block_counts = Counter(block_type for chunk in chunks for block_type in chunk["block_types"])
    for block_type, count in block_counts.most_common():
        print(f"  {block_type:<10} {count}")

    merged = sum(len(chunk["section_ids"]) > 1 for chunk in chunks)
    print(f"\nChunks built from more than one section: {merged}")

    by_size = sorted(chunks, key=lambda chunk: chunk["token_count"])
    print("\nSmallest chunks")
    for chunk in by_size[:MAX_EXAMPLES]:
        print(f"  {chunk['token_count']:>4}  {chunk['chunk_id']}")
    print("Largest chunks")
    for chunk in by_size[-MAX_EXAMPLES:]:
        print(f"  {chunk['token_count']:>4}  {chunk['chunk_id']}")

    print("\nDocuments with most chunks")
    for url, count in Counter(chunk["source_url"] for chunk in chunks).most_common(MAX_EXAMPLES):
        print(f"  {count:>4}  {url}")


# ---------------------------------------------------------
# Printing chunks
# ---------------------------------------------------------


def print_chunk(chunk: dict[str, Any], max_chars: int):
    print("\n" + "=" * 70)
    print(f"chunk_id:     {chunk['chunk_id']}")
    print(f"source_url:   {chunk['source_url']}")
    print(f"breadcrumb:   {' > '.join(chunk['breadcrumb']) or '-'}")
    print(f"heading_path: {' > '.join(chunk['heading_path'])}")
    print(f"section_ids:  {chunk['section_ids']}")
    print(f"block_types:  {chunk['block_types']}   tokens: {chunk['token_count']}")
    print("-" * 70)

    text = chunk["text"]
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + f"\n... [{len(chunk['text']) - max_chars} more chars]"
    print(text)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():

    parser = argparse.ArgumentParser(description="Validate and inspect chunks.jsonl")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sections", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--samples", type=int, default=5, help="random chunks to print")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--doc", help="print all chunks whose source_url contains this text")
    parser.add_argument("--max-chars", type=int, default=1500, help="truncate printed chunks (0 = full)")
    args = parser.parse_args()

    with args.chunks.open(encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file]

    print(f"Loaded {len(chunks)} chunks from {args.chunks}")

    print_problems("Validation", validate(chunks))
    print_problems("Coverage vs sections.jsonl", check_coverage(chunks, args.sections))
    print_stats(chunks)

    if args.doc:
        selected = [chunk for chunk in chunks if args.doc in chunk["source_url"]]
        print(f"\nAll {len(selected)} chunks matching '{args.doc}'")
    else:
        selected = random.Random(args.seed).sample(chunks, min(args.samples, len(chunks)))
        print(f"\n{len(selected)} random sample chunks (seed {args.seed})")

    for chunk in selected:
        print_chunk(chunk, args.max_chars)


if __name__ == "__main__":
    main()
