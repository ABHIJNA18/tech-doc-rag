# ---------------------------------------------------------
# preprocess_docs.py
#
# Converts the raw IONOS documentation Markdown files into
# structured sections, one record per heading section.
#
# Input:  ~/Projects/ionos-docs/raw/**/*.md   (downloaded pages)
# Output: ~/Projects/ionos-docs/processed/sections.jsonl
#
# Steps:
#   1. Clean GitBook syntax (navigation boilerplate, tabs, hints,
#      content-refs, steppers, space.vars.* variables).
#   2. Parse the Markdown with markdown-it-py.
#   3. Split the page into sections at headings, keeping the
#      heading path, source URL and a stable section ID, plus a
#      breadcrumb of parent page titles from llms.txt (so pages
#      titled "FAQ" or "Models" still say which product they cover).
#   4. Keep paragraphs, lists, tables and code blocks as readable
#      text. Embedded OpenAPI specs (API reference pages) are
#      converted into readable Markdown instead of raw JSON.
#
# Usage:
#   python scripts/preprocess_docs.py
# ---------------------------------------------------------

from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

RAW_DIR = Path.home() / "Projects" / "ionos-docs" / "raw"
PROCESSED_DIR = Path.home() / "Projects" / "ionos-docs" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "sections.jsonl"

# Documentation index downloaded by download_raw_docs.py. Used for
# page titles of parent pages (breadcrumbs).
LLMS_INDEX = Path(__file__).resolve().parent / "llms.txt"


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

    # The raw Markdown often escapes underscores in variable names
    # (space.vars.ionos\_cloud\_ai\_model\_hub). Unescape them first,
    # otherwise the short "space.vars.ionos" rule below matches and
    # leaves "_cloud_ai_model_hub" behind.
    text = re.sub(
        r"space\.vars\.[A-Za-z0-9\\_]+",
        lambda m: m.group(0).replace("\\_", "_"),
        text,
    )

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

BREAK_TAG_RE = re.compile(r"<br\s*/?>|</(?:p|li)>", re.IGNORECASE)


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
            # Line-break tags (<br>, </p>, </li>) separate text,
            # so they become a space instead of nothing.
            html = BREAK_TAG_RE.sub(" ", child.content)

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
# OpenAPI specs
#
# API reference pages embed their OpenAPI spec as one large JSON
# code block, which GitBook renders as the endpoint or model page.
# We convert that JSON into readable Markdown with the same facts
# (method, path, servers, parameters, responses, fields) instead
# of keeping raw JSON. Nothing is generated or summarised: every
# line comes from a field in the spec.
#
# - Endpoint pages: method + path, servers, authentication,
#   parameters, request body fields and responses. The request body
#   is expanded to its full depth, like GitBook's Body section.
# - Models pages ("The X object"): only schema X. Each block also
#   contains every schema X references, which have their own
#   sections, so referenced objects are named, not repeated.
# ---------------------------------------------------------

HTTP_METHODS = ("get", "post", "put", "patch", "delete", "head", "options")

# Safety limit for nested fields. Real specs nest at most 5 levels;
# objects that refer to themselves are stopped separately.
MAX_FIELD_DEPTH = 10

MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")

MARKDOWN_HEADING_RE = re.compile(r"^\s*#+\s+(.*?)\s*$")

# Bold written as __Note__ (not identifiers such as __init__.py).
MARKDOWN_UNDERSCORE_BOLD_RE = re.compile(r"(?<![\w.])__(\w[^_\n]*?)__(?![\w.])")


def parse_openapi(code: str) -> dict[str, Any] | None:
    code = code.strip()

    if not code.startswith("{") or '"openapi"' not in code[:100]:
        return None

    try:
        spec = json.loads(code)
    except json.JSONDecodeError:
        return None

    return spec if isinstance(spec, dict) and "openapi" in spec else None


def plain_text(text: str) -> str:
    """
    Markdown inside spec descriptions, reduced the same way
    inline_text() reduces page text: links keep their text,
    emphasis and code markers are dropped.
    """

    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = text.replace("**", "").replace("`", "")
    text = MARKDOWN_UNDERSCORE_BOLD_RE.sub(r"\1", text)

    # "### Bad Request" -> "Bad Request." so it reads as a lead-in
    # sentence when description lines are joined.
    lines = []
    for line in text.splitlines():
        match = MARKDOWN_HEADING_RE.match(line)
        if match:
            line = match.group(1)
            if line and line[-1] not in ".:!?":
                line += "."
        lines.append(line)

    return "\n".join(lines)


def spec_description(text: str | None) -> str:
    """
    Full description as Markdown paragraphs. Prose lines are joined,
    tables inside the description are kept as tables.
    """

    blocks: list[str] = []
    prose: list[str] = []
    table: list[str] = []

    def flush():
        if prose:
            blocks.append(" ".join(prose))
            prose.clear()
        if table:
            blocks.append("\n".join(table))
            table.clear()

    for line in plain_text(text or "").splitlines():

        # Blockquote markers (> Note ...) carry no content.
        line = re.sub(r"^\s*>\s?", "", line).strip()

        if not line:
            flush()
        elif line.startswith("|"):
            if prose:
                flush()
            table.append(line)
        else:
            if table:
                flush()
            prose.append(line)

    flush()

    return "\n\n".join(re.sub(r"[ \t]+", " ", block) for block in blocks)


def short_description(text: str | None) -> str:
    """
    One-line description for a list item. Stops before any table,
    because a table cannot live inside a list line.
    """

    lines = []

    for line in plain_text(text or "").splitlines():
        if line.lstrip().startswith("|"):
            break
        lines.append(re.sub(r"^\s*>\s?", "", line))

    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def value_table(text: str | None) -> list[tuple[str, str]]:
    """
    (value, meaning) pairs from a table inside a description, e.g.
    the eviction policy table: ("allkeys-lru", "The least recently
    used keys will be removed first.").
    """

    rows = [line.strip() for line in plain_text(text or "").splitlines() if line.strip().startswith("|")]

    pairs = []

    # rows[0] is the header row, rows[1] the "| --- |" separator.
    for row in rows[2:]:
        cells = [cell.strip() for cell in row.strip("|").split("|")]

        if len(cells) >= 2 and cells[0]:
            pairs.append((cells[0], re.sub(r"\s+", " ", cells[1])))

    return pairs


def resolve(spec: dict, node: Any) -> tuple[str | None, dict]:
    """
    Follow "$ref": "#/components/schemas/Name" references.
    Returns the referenced name (if any) and the target node.
    """

    name = None

    for _ in range(10):
        if not isinstance(node, dict) or "$ref" not in node:
            break

        ref = node["$ref"]
        name = ref.rsplit("/", 1)[-1]

        target: Any = spec
        for part in ref.lstrip("#/").split("/"):
            target = target.get(part, {}) if isinstance(target, dict) else {}

        node = target

    return name, node if isinstance(node, dict) else {}


def is_object(schema: dict) -> bool:
    # "additionalProperties" alone describes a free-form object.
    return schema.get("type") == "object" or any(
        key in schema for key in ("properties", "allOf", "additionalProperties")
    )


def type_label(spec: dict, schema: dict) -> str:
    name, target = resolve(spec, schema)

    if name and is_object(target):
        return f"{name} object"

    options = target.get("oneOf") or target.get("anyOf")
    if options:
        return " or ".join(type_label(spec, option) for option in options)

    schema_type = target.get("type")

    if schema_type == "array":
        return f"array of {type_label(spec, target.get('items', {}))}"

    if isinstance(schema_type, list):
        schema_type = " or ".join(schema_type)

    label = schema_type or ("object" if is_object(target) else "any")

    if target.get("format"):
        label += f", {target['format']}"

    return label


def value_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def constraint_facts(schema: dict) -> list[str]:
    """
    Limits and allowed values, each as an explicit "label: value"
    (the same labels GitBook shows: default, min, max, ...).
    """

    facts = []

    if "enum" in schema:
        facts.append("possible values: " + ", ".join(value_text(v) for v in schema["enum"]))

    for key, label in (
        ("default", "default"),
        ("minimum", "min"),
        ("maximum", "max"),
        ("minLength", "min length"),
        ("maxLength", "max length"),
        ("minItems", "min items"),
        ("maxItems", "max items"),
        ("pattern", "pattern"),
    ):
        if key in schema:
            facts.append(f"{label}: {value_text(schema[key])}")

    for flag, label in (
        ("readOnly", "read-only"),
        ("writeOnly", "write-only"),
        ("nullable", "nullable"),
        ("deprecated", "deprecated"),
    ):
        if schema.get(flag):
            facts.append(label)

    return facts


def object_fields(spec: dict, schema: dict) -> tuple[dict, set]:
    """
    Properties and required names of an object, merging allOf parts.
    """

    properties: dict = {}
    required: set = set()

    for part in schema.get("allOf", []):
        _, part = resolve(spec, part)
        part_properties, part_required = object_fields(spec, part)
        properties.update(part_properties)
        required |= part_required

    properties.update(schema.get("properties", {}))
    required |= set(schema.get("required", []))

    return properties, required


def field_lines(
    spec: dict,
    schema: dict,
    request_body: bool,
    depth: int = 0,
    seen: frozenset = frozenset(),
) -> list[str]:
    """
    One list item per field, with nested objects as indented items.

    Models pages (request_body=False): only inline objects are
    expanded; referenced objects are named, because they have their
    own section.

    Request bodies (request_body=True), like GitBook's Body section:
      - all nested objects are expanded, to full depth
      - read-only fields are listed and labelled "read-only" (the
        server sets them), but read-only objects are not expanded
      - tables of allowed values in a description become nested
        items ("allkeys-lru: The least recently used keys ...")
    """

    properties, required = object_fields(spec, schema)

    lines = []

    for name, prop in properties.items():

        ref_name, target = resolve(spec, prop)

        # Keys next to a $ref (e.g. readOnly, description) apply too.
        merged = {**target, **{k: v for k, v in prop.items() if k != "$ref"}}

        read_only = bool(merged.get("readOnly"))

        facts = [type_label(spec, prop)]
        facts.append("required" if name in required else "optional")
        facts += constraint_facts(merged)

        # Facts are separated by ";" because a list of possible
        # values already uses commas.
        line = "  " * depth + f"- {name} ({'; '.join(facts)})"

        description = short_description(merged.get("description"))
        if description:
            line += f": {description}"

        lines.append(line)

        if request_body:
            for value, meaning in value_table(merged.get("description")):
                lines.append("  " * (depth + 1) + f"- {value}: {meaning}")

        nested_name, nested = ref_name, target
        if target.get("type") == "array":
            nested_name, nested = resolve(spec, target.get("items", {}))

        # "seen" holds the named objects already open on this branch,
        # so an object that refers to itself is not expanded forever.
        if (
            is_object(nested)
            and depth + 1 < MAX_FIELD_DEPTH
            and (request_body or nested_name is None)
            and not (request_body and read_only)
            and nested_name not in seen
        ):
            lines += field_lines(
                spec,
                nested,
                request_body,
                depth + 1,
                seen | {nested_name} if nested_name else seen,
            )

    return lines


def api_line(spec: dict) -> str:
    info = spec.get("info", {})
    title = info.get("title", "API")
    version = info.get("version")

    return f"API: {title}" + (f" (version {version})" if version else "")


def auth_text(spec: dict, operation: dict) -> str:
    schemes = spec.get("components", {}).get("securitySchemes", {})
    requirements = operation.get("security", spec.get("security", []))

    parts = []

    for requirement in requirements:
        for name in requirement:
            scheme = schemes.get(name, {})
            kind = scheme.get("type")

            if kind == "http" and scheme.get("scheme") == "bearer":
                text = "Bearer token in the Authorization header"
                if scheme.get("bearerFormat"):
                    text += f" ({scheme['bearerFormat']})"
            elif kind == "http":
                text = f"HTTP {scheme.get('scheme', '')} authentication".replace("  ", " ")
            elif kind == "apiKey":
                text = f"API key in the {scheme.get('name')} {scheme.get('in')}"
            else:
                text = name

            description = short_description(scheme.get("description"))
            parts.append(f"{text}. {description}" if description else f"{text}.")

    return "Authentication: " + " ".join(parts) if parts else ""


def parameter_line(spec: dict, parameter: dict) -> str:
    schema = parameter.get("schema", {})
    _, target = resolve(spec, schema)

    facts = [type_label(spec, schema)]
    facts.append("required" if parameter.get("required") else "optional")
    facts += constraint_facts({**target, **{k: v for k, v in schema.items() if k != "$ref"}})

    line = f"- {parameter.get('name')} ({'; '.join(facts)})"

    description = short_description(parameter.get("description") or target.get("description"))

    return f"{line}: {description}" if description else line


def media_schema(content: dict) -> dict | None:
    if not content:
        return None

    media = content.get("application/json") or next(iter(content.values()))

    return media.get("schema")


def openapi_endpoint_markdown(spec: dict) -> str:
    parts = [api_line(spec)]

    for path, path_item in spec.get("paths", {}).items():
        for method, operation in path_item.items():

            if method not in HTTP_METHODS:
                continue

            parts.append(f"{method.upper()} {path}")

            servers = operation.get("servers") or spec.get("servers", [])
            if servers:
                parts.append("Servers:")
                parts.append(
                    "\n".join(
                        f"- {server['url']}"
                        + (f" ({short_description(server['description'])})" if server.get("description") else "")
                        for server in servers
                    )
                )

            auth = auth_text(spec, operation)
            if auth:
                parts.append(auth)

            # Parameters, grouped by location (path, query, header).
            parameters = [
                resolve(spec, parameter)[1]
                for parameter in path_item.get("parameters", []) + operation.get("parameters", [])
            ]

            for location in ("path", "query", "header", "cookie"):
                lines = [parameter_line(spec, p) for p in parameters if p.get("in") == location]
                if lines:
                    parts.append(f"{location.capitalize()} parameters:")
                    parts.append("\n".join(lines))

            # Request body, with its fields expanded.
            _, body = resolve(spec, operation.get("requestBody"))
            body_schema = media_schema(body.get("content", {}))

            if body_schema is not None:
                _, body_target = resolve(spec, body_schema)

                text = f"Request body ({type_label(spec, body_schema)}"
                text += ", required)" if body.get("required") else ")"

                # The description often sits on the body's object
                # rather than on the request body itself.
                description = short_description(body.get("description") or body_target.get("description"))
                parts.append(f"{text}: {description}" if description else text)

                lines = field_lines(spec, body_target, request_body=True)
                if lines:
                    parts.append("Request body fields:")
                    parts.append("\n".join(lines))

            # Responses: status code, description and body type.
            lines = []
            for code, response in operation.get("responses", {}).items():
                _, response = resolve(spec, response)
                line = f"- {code}: {short_description(response.get('description')) or 'No description.'}"

                if line[-1] not in ".:!?":
                    line += "."

                response_schema = media_schema(response.get("content", {}))
                if response_schema is not None:
                    line += f" Returns: {type_label(spec, response_schema)}."

                lines.append(line)

            if lines:
                parts.append("Responses:")
                parts.append("\n".join(lines))

    return "\n\n".join(parts)


def openapi_model_markdown(spec: dict, heading: str) -> str | None:
    schemas = spec.get("components", {}).get("schemas", {})

    if not schemas:
        return None

    # Models page headings look like "The ReplicaSet object".
    match = re.fullmatch(r"The (.+) object", heading or "")
    name = match.group(1) if match and match.group(1) in schemas else next(iter(schemas))
    schema = schemas[name]

    parts = [api_line(spec)]

    description = spec_description(schema.get("description"))
    if description:
        parts.append(description)

    if is_object(schema):
        lines = field_lines(spec, schema, request_body=False)
        if lines:
            parts.append("Fields:")
            parts.append("\n".join(lines))
    else:
        # A simple value (string, integer, enum): one fact per line,
        # e.g. "Type: string (enum)", "Default: allkeys-lru".
        lines = [f"Type: {type_label(spec, schema)}" + (" (enum)" if "enum" in schema else "")]
        lines += [fact[0].upper() + fact[1:] for fact in constraint_facts(schema)]
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def openapi_to_markdown(spec: dict, heading: str) -> str | None:
    has_operations = any(
        method in HTTP_METHODS
        for path_item in spec.get("paths", {}).values()
        for method in path_item
    )

    if has_operations:
        return openapi_endpoint_markdown(spec)

    return openapi_model_markdown(spec, heading)


# ---------------------------------------------------------
# Add Source URL
# ---------------------------------------------------------

def source_url(source_path: Path) -> str:
    relative_path = source_path.relative_to(RAW_DIR)
    relative_path = relative_path.with_suffix("")

    return f"https://docs.ionos.com/cloud/{relative_path.as_posix()}"


# ---------------------------------------------------------
# Breadcrumb
#
# Many pages have generic titles ("FAQ", "Overview", "Error Codes",
# "Models") that do not say which product they belong to. The
# breadcrumb is the list of titles of the page's parent pages, taken
# from the documentation index, e.g.
#   .../ai/ai-model-hub/error-codes  ->  ["AI Model Hub"]
# ---------------------------------------------------------

LLMS_LINK_RE = re.compile(r"\[([^\]]+)\]\((https://docs\.ionos\.com/cloud/[^)\s]+?)\.md\)")


def load_page_titles(index_path: Path) -> dict[str, str]:
    """
    {page URL (without .md): page title} from llms.txt.
    """

    if not index_path.exists():
        return {}

    titles: dict[str, str] = {}

    for match in LLMS_LINK_RE.finditer(index_path.read_text(encoding="utf-8")):
        titles.setdefault(match.group(2), match.group(1))

    return titles


def breadcrumb(page_url: str, titles: dict[str, str]) -> list[str]:
    """
    Titles of the parent pages of page_url, outermost first. Parent
    paths that are not pages themselves (e.g. /cloud/ai) are skipped.
    """

    base, _, path = page_url.partition("/cloud/")
    parts = path.split("/")

    crumbs = []

    for depth in range(1, len(parts)):
        title = titles.get(f"{base}/cloud/{'/'.join(parts[:depth])}")
        if title:
            crumbs.append(title)

    return crumbs


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
    page_breadcrumb: list[str] | None = None,
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
                "breadcrumb": list(page_breadcrumb or []),
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

            # Embedded OpenAPI specs become readable Markdown
            # (see "OpenAPI specs" above); other code is kept as is.
            spec = parse_openapi(token.content)
            heading = heading_stack[-1] if heading_stack else title
            api_markdown = openapi_to_markdown(spec, heading) if spec else None

            if api_markdown:
                block_type = "api_spec"
                content = api_markdown
            else:
                block_type = "code"
                content = token_to_text(token)

            if content:

                current_blocks.append(
                    {
                        "block_type": block_type,
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

            # Entities such as &#x26; or &amp; become "&".
            visible = unescape(visible)

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

    page_titles = load_page_titles(LLMS_INDEX)

    if not page_titles:
        print(f"WARNING: {LLMS_INDEX} not found or empty; sections get no breadcrumb.")

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
                    breadcrumb(source_url(path), page_titles),
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
