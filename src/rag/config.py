# ---------------------------------------------------------
# config.py
#
# Central settings for the RAG system: data locations, model names
# and the Chroma collection name.
#
# Every value has a local default and can be overridden with an
# environment variable (e.g. in .env, or on a server when deployed),
# so no path or key is hard-coded in the rest of the code.
# Secrets such as OPENAI_API_KEY are never stored here; they are
# read from the environment by the libraries that need them.
# ---------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Repository root: src/rag/config.py -> parents[2]
REPO_DIR = Path(__file__).resolve().parents[2]

# Load .env from the repository root (git-ignored). Variables that are
# already set in the environment take precedence.
load_dotenv(REPO_DIR / ".env")


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


# ---------------------------------------------------------
# Data locations (outside the repository)
# ---------------------------------------------------------

DATA_DIR = _path("RAG_DATA_DIR", Path.home() / "Projects" / "ionos-docs")

CHUNKS_FILE = _path("RAG_CHUNKS_FILE", DATA_DIR / "processed" / "chunks.jsonl")
CHROMA_DIR = _path("RAG_CHROMA_DIR", DATA_DIR / "chroma")
EMBEDDING_CACHE_DIR = _path("RAG_EMBEDDING_CACHE_DIR", DATA_DIR / "embedding_cache")


# ---------------------------------------------------------
# Index and models
# ---------------------------------------------------------

COLLECTION_NAME = os.environ.get("RAG_COLLECTION_NAME", "ionos_docs")
EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "text-embedding-3-small")


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

# Which retrieval method to use by default ("vector" for now; phase 2
# adds e.g. "bm25", "hybrid", "hybrid_rerank"). See rag/retrieval.py.
RETRIEVAL_MODE = os.environ.get("RAG_RETRIEVAL_MODE", "vector")

# How many chunks to retrieve per question.
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
