from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

RAW_DIR = Path.home() / "Projects" / "ionos-docs" / "raw"
PROCESSED_DIR = Path.home() / "Projects" / "ionos-docs" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "sections.jsonl"


# ---------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------

md = MarkdownIt("commonmark").enable("table")


# ---------------------------------------------------------
# GitBook cleanup
# ---------------------------------------------------------


def clean_gitbook_markdown(text: str) -> str:
    """
    Remove GitBook-specific syntax while preserving
    the actual documentation content.
    """

    # Remove GitBook-generated page navigation boilerplate.
    text = re.sub(
        r"^>\s*For the complete documentation index,.*?"
        r"this page is available as.*?\.\s*$",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # -----------------------------------------------------
    # Remove everything after Agent Instructions.
    #
    # These are instructions for AI agents, not part of
    # the actual IONOS documentation.
    # -----------------------------------------------------

    text = re.split(
        r"\n---\s*\n# Agent Instructions",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    # -----------------------------------------------------
    # GitBook content blocks.
    #
    # Keep their contents, remove only the wrapper.
    # -----------------------------------------------------

    text = re.sub(
        r"{%\s*(?:end)?content-ref[^%]*%}",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"{%\s*(?:end)?stepper\s*%}",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"{%\s*(?:end)?step\s*%}",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # -----------------------------------------------------
    # GitBook tabs.
    #
    # Keep the content inside tabs, remove only the wrapper.
    # -----------------------------------------------------

    text = re.sub(
        r"{%\s*tabs\s*%}",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r'{%\s*tab\s+title="[^"]*"\s*%}',
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"{%\s*endtab\s*%}",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"{%\s*endtabs\s*%}",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r'{%\s*file\s+src="[^"]*"\s*%}',
        "",
        text,
        flags=re.IGNORECASE,
    )

    # -----------------------------------------------------
    # GitBook hints.
    #
    # Keep the text inside the hint.
    # -----------------------------------------------------

    text = re.sub(
        r"{%\s*hint[^%]*%}",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"{%\s*endhint\s*%}",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # -----------------------------------------------------
    # GitBook expression variables.
    #
    # Example:
    #
    # <code class="expression">space.vars.ionos_cloud</code>
    #
    # becomes:
    #
    # IONOS CLOUD
    # -----------------------------------------------------

    text = re.sub(
        r'<code\s+class="expression">\s*' r"space\.vars\.ionos\\?_cloud" r"\s*</code>",
        "IONOS CLOUD",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"space\.vars\.ionos_cloud_ai_model_hub", "IONOS AI Model Hub", text, flags=re.IGNORECASE)
    text = re.sub(r"space\.vars\.ionos_cloud_object_storage", "IONOS CLOUD Object Storage", text, flags=re.IGNORECASE)
    text = re.sub(r"space\.vars\.ionos_cloud_apis?", "IONOS CLOUD API", text, flags=re.IGNORECASE)
    text = re.sub(r"space\.vars\.backup_service", "Backup Service", text, flags=re.IGNORECASE)
    text = re.sub(r"space\.vars\.backup\b", "Backup", text, flags=re.IGNORECASE)
    text = re.sub(r"space\.vars\.ionos\b", "IONOS", text, flags=re.IGNORECASE)

    # -----------------------------------------------------
    # Any remaining <code>...</code> tags.
    #
    # Keep their contents.
    # -----------------------------------------------------

    text = re.sub(
        r"<code[^>]*>(.*?)</code>",
        lambda m: f" {m.group(1)} ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # -----------------------------------------------------
    # GitBook <mark> formatting.
    #
    # Keep text, remove HTML wrapper.
    # -----------------------------------------------------

    text = re.sub(
        r"</?mark[^>]*>",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # -----------------------------------------------------
    # Normalize escaped underscores.
    # -----------------------------------------------------

    text = text.replace(r"\_", "_")

    # -----------------------------------------------------
    # Normalize excessive blank lines.
    # -----------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ---------------------------------------------------------
# Markdown utilities
# ---------------------------------------------------------


def inline_text(token) -> str:
    """
    Extract readable text from an inline Markdown token
    while preserving whitespace from the parsed Markdown.
    """

    if not token.children:
        return token.content

    parts = []

    for child in token.children:

        if child.type == "text":
            parts.append(child.content)

        elif child.type == "code_inline":
            parts.append(child.content)

        elif child.type == "image":
            alt = child.content.strip()

            if alt:
                parts.append(f"[Image: {alt}]")

        elif child.type in {"softbreak", "hardbreak"}:
            parts.append("\n")

        elif child.type == "html_inline":
            # GitBook HTML wrappers such as <code>...</code>.
            # Keep the visible text but remove the HTML tags.
            html = child.content

            visible = re.sub(r"<[^>]+>", "", html)
            parts.append(visible)

        # link_open / link_close / emphasis_open / emphasis_close /
        # strong_open / strong_close etc. carry no visible text,
        # so they are intentionally ignored.

    text = "".join(parts)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()

    
def token_to_text(token) -> str:
    """
    Convert a Markdown token into readable text.
    """

    if token.type == "inline":
        return inline_text(token).strip()

    if token.type == "fence":
        language = token.info.strip()

        if language:
            return f"```{language}\n{token.content.rstrip()}\n```"

        return f"```\n{token.content.rstrip()}\n```"

    if token.type == "code_block":
        return f"```\n{token.content.rstrip()}\n```"

    return token.content.strip()

# ---------------------------------------------------------
# Table handling
# ---------------------------------------------------------


def table_to_markdown(tokens: list, start_index: int) -> tuple[str, int]:
    """
    Convert a Markdown table token sequence into a readable
    Markdown table.

    Returns:
        (table_text, next_index)
    """

    rows: list[list[str]] = []

    i = start_index

    while i < len(tokens):

        token = tokens[i]

        if token.type == "table_close":
            break

        if token.type == "tr_open":
            cells = []

            j = i + 1

            while j < len(tokens):

                cell = tokens[j]

                if cell.type in {
                    "th_close",
                    "td_close",
                }:
                    j += 1
                    continue

                if cell.type == "inline":
                    cells.append(inline_text(cell).strip())

                if cell.type == "tr_close":
                    break

                j += 1

            if cells:
                rows.append(cells)

        i += 1

    if not rows:
        return "", i

    # Normalize column count.
    column_count = max(len(row) for row in rows)

    normalized = [row + [""] * (column_count - len(row)) for row in rows]

    lines = []

    # Header
    lines.append("| " + " | ".join(normalized[0]) + " |")

    lines.append("| " + " | ".join(["---"] * column_count) + " |")

    # Body
    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines), i

# ---------------------------------------------------------
# Add Source URL
# ---------------------------------------------------------

def source_url(source_path: Path) -> str:
    relative_path = source_path.relative_to(RAW_DIR)
    relative_path = relative_path.with_suffix("")

    return f"https://docs.ionos.com/cloud/{relative_path.as_posix()}"


# ---------------------------------------------------------
# Unique Section ID
# ---------------------------------------------------------

def document_id(source_path: Path) -> str:
    relative_path = source_path.relative_to(RAW_DIR)
    return str(relative_path.with_suffix("")).replace("/", ":")


# ---------------------------------------------------------
# Parse document
# ---------------------------------------------------------


def parse_document(
    text: str,
    source_path: Path,
) -> dict[str, Any]:

    cleaned = clean_gitbook_markdown(text)

    tokens = md.parse(cleaned)

    title = None

    sections: list[dict[str, Any]] = []

    heading_stack: list[str] = []

    current_blocks: list[dict[str, Any]] = []

    section_counter = 0

    def flush_section():
        nonlocal current_blocks
        nonlocal section_counter

        if not current_blocks:
            return

        content_parts = []

        block_types = []

        for block in current_blocks:

            content = block["content"].strip()

            if not content:
                continue

            content_parts.append(content)
            block_types.append(block["block_type"])

        content = "\n\n".join(content_parts).strip()

        if not content:
            current_blocks = []
            return

        section_counter += 1

        sections.append(
            {
                "section_id": f"{document_id(source_path)}:section:{section_counter}",
                "document_title": title,
                "source_file": str(source_path),
                "source_url": source_url(source_path),
                "heading": (heading_stack[-1] if heading_stack else title),
                "heading_path": heading_stack.copy(),
                "block_types": sorted(set(block_types)),
                "content": content,
            }
        )

        current_blocks = []

    i = 0

    while i < len(tokens):

        token = tokens[i]

        # -------------------------------------------------
        # Headings
        # -------------------------------------------------

        if token.type == "heading_open":

            level = int(token.tag[1])

            heading_token = tokens[i + 1]

            heading = inline_text(
                heading_token
            ).strip()

            # Use the actual H1 as the document title.
            if level == 1 and title is None:
                title = heading


            flush_section()

            heading_stack = heading_stack[: level - 1]
            heading_stack.append(heading)

            i += 3
            continue

        # -------------------------------------------------
        # Paragraphs
        # -------------------------------------------------

        if token.type == "paragraph_open":

            inline_token = tokens[i + 1]

            content = inline_text(inline_token).strip()

            if content:

                current_blocks.append(
                    {
                        "block_type": "paragraph",
                        "content": content,
                    }
                )

            i += 3
            continue

        # -------------------------------------------------
        # Lists
        # -------------------------------------------------

        if token.type in {
            "bullet_list_open",
            "ordered_list_open",
        }:

            list_items = []

            list_type = "ordered" if token.type == "ordered_list_open" else "unordered"

            j = i + 1

            while j < len(tokens):

                if tokens[j].type in {
                    "bullet_list_close",
                    "ordered_list_close",
                }:
                    break

                if tokens[j].type == "inline":

                    item = inline_text(tokens[j]).strip()

                    if item:
                        list_items.append(item)

                j += 1

            if list_items:

                prefix = "1." if list_type == "ordered" else "-"

                content = "\n".join(f"{prefix} {item}" for item in list_items)

                current_blocks.append(
                    {
                        "block_type": "list",
                        "content": content,
                    }
                )

            i = j + 1
            continue

        # -------------------------------------------------
        # Code blocks
        # -------------------------------------------------

        if token.type in {
            "fence",
            "code_block",
        }:

            content = token_to_text(token)

            if content:

                current_blocks.append(
                    {
                        "block_type": "code",
                        "content": content,
                    }
                )

            i += 1
            continue

        # -------------------------------------------------
        # Tables
        # -------------------------------------------------

        if token.type == "table_open":

            table, next_index = table_to_markdown(
                tokens,
                i + 1,
            )

            if table:

                current_blocks.append(
                    {
                        "block_type": "table",
                        "content": table,
                    }
                )

            i = next_index + 1
            continue

        # -------------------------------------------------
        # HTML blocks
        #
        # Some GitBook pages contain HTML cards.
        # We preserve their readable text rather than
        # throwing it away.
        # -------------------------------------------------

        if token.type == "html_block":

            html = token.content.strip()

            # Remove tags but keep visible text.
            visible = re.sub(
                r"<[^>]+>",
                " ",
                html,
            )

            visible = re.sub(
                r"\s+",
                " ",
                visible,
            ).strip()

            if visible:

                current_blocks.append(
                    {
                        "block_type": "html",
                        "content": visible,
                    }
                )

            i += 1
            continue

        i += 1

    # Flush final section.
    flush_section()

    if title is None:
        title = source_path.stem.replace("-", " ").title()
    
    return {
        "document_title": title,
        "source_file": str(source_path),
        "sections": sections,
    }


# ---------------------------------------------------------
# Process all documents
# ---------------------------------------------------------


def main():

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw directory not found: {RAW_DIR}")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(RAW_DIR.rglob("*.md"))

    print(f"Found {len(files)} Markdown files.")

    total_sections = 0

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as output:

        for path in files:

            try:

                text = path.read_text(encoding="utf-8")

                document = parse_document(
                    text,
                    path,
                )

                for section in document["sections"]:

                    output.write(
                        json.dumps(
                            section,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    total_sections += 1

                print(
                    f"OK  {path.relative_to(RAW_DIR)} "
                    f"→ {len(document['sections'])} sections"
                )

            except Exception as exc:

                print(f"FAIL {path}: {exc}")

    print()
    print("=" * 50)
    print("Preprocessing complete")
    print("=" * 50)
    print(f"Documents: {len(files)}")
    print(f"Sections:  {total_sections}")
    print(f"Output:    {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
