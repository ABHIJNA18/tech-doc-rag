# ---------------------------------------------------------
# documents.py
#
# Conversion between our chunk records (chunks.jsonl, written by
# scripts/chunk_sections.py) and LangChain Documents.
#
# load_chunks(path) reads chunks.jsonl into a list of chunk records.
# chunk_to_document(chunk) turns one chunk into a Document:
#
# A LangChain Document has two parts:
#   - page_content: the text that is embedded and later shown to
#     the LLM. For us: the chunk text, including its context header
#     ("AI Model Hub > Error Codes > API errors").
#   - metadata: stored next to the vector, not embedded. Used for
#     citations (source URL, breadcrumb, section) and filtering.
#
# Chroma only accepts simple metadata values (str, int, float,
# bool), so list fields are stored as JSON strings and turned back
# into lists by document_to_chunk().
# ---------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

# Chunk fields kept as metadata. "text" becomes page_content.
SCALAR_FIELDS = ["chunk_id", "chunk_index", "document_title", "source_url", "token_count"]
LIST_FIELDS = ["breadcrumb", "heading_path", "section_ids", "block_types"]


def load_chunks(path: Path) -> list[dict[str, Any]]:
    """
    Read chunks.jsonl: one chunk record per line.
    """

    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def chunk_to_document(chunk: dict[str, Any]) -> Document:
    """
    One chunk record -> one LangChain Document.
    """

    metadata = {field: chunk[field] for field in SCALAR_FIELDS}
    metadata.update({field: json.dumps(chunk[field], ensure_ascii=False) for field in LIST_FIELDS})
    
    #lists stored as JSON text because Chroma accepts only simple values.
    return Document(id=chunk["chunk_id"], page_content=chunk["text"], metadata=metadata)


def document_to_chunk(document: Document) -> dict[str, Any]:
    """
    The reverse: a Document (e.g. returned by a Chroma search) back
    into a chunk record with real lists, for citations.
    """

    chunk = {field: document.metadata[field] for field in SCALAR_FIELDS}
    chunk.update({field: json.loads(document.metadata[field]) for field in LIST_FIELDS})
    chunk["text"] = document.page_content

    return chunk
