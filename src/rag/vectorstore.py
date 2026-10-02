# ---------------------------------------------------------
# vectorstore.py
#
# The embedding model and the Chroma vector store, set up in one
# place so that indexing (scripts/build_index.py) and retrieval use
# exactly the same model, collection and distance metric.
#
# - make_embeddings(): OpenAI embeddings wrapped in a disk cache.
#   The cache stores every chunk vector keyed by a hash of its text,
#   so rebuilding the index only pays for new or changed chunks.
# - chroma_client(): Chroma in embedded mode: a folder on disk, no
#   separate database server.
# - open_vector_store(): the LangChain Chroma wrapper for our
#   collection (cosine similarity).
# ---------------------------------------------------------

from __future__ import annotations

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from rag import config

DISTANCE = "cosine"


def make_embeddings(cached: bool = True) -> Embeddings:
    """
    The embedding model. With cached=True, chunk vectors are saved in
    EMBEDDING_CACHE_DIR; the namespace (model name) makes sure vectors
    of a different model are never reused. Question embeddings are
    not cached.
    """

    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

    if not cached:
        return embeddings

    store = LocalFileStore(str(config.EMBEDDING_CACHE_DIR))

    return CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        store,
        namespace=config.EMBEDDING_MODEL,
        key_encoder="sha256",
    )


def chroma_client() -> chromadb.ClientAPI:
    """
    Chroma reading/writing the folder CHROMA_DIR. Anonymous usage
    statistics are switched off.
    """

    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def open_vector_store(embeddings: Embeddings, client: chromadb.ClientAPI | None = None) -> Chroma:
    """
    Our collection as a LangChain vector store. Created if missing.
    """

    return Chroma(
        client=client or chroma_client(),
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        collection_configuration={"hnsw": {"space": DISTANCE}},
    )
