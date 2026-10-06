# ---------------------------------------------------------
# retrieval.py
#
# Finding the chunks that are most relevant to a question.
#
# Retrieval methods are interchangeable: each one is a class with a
# `name` and a `search(question, k)` method returning RetrievedChunk
# objects, registered in RETRIEVERS under its name. The rest of the
# system (generation, API, evaluation) only calls `retrieve()`, which
# picks a method by name. That lets the evaluation compare methods
# one at a time, e.g.:
#
#     retrieve(question, mode="vector")         # phase 1 baseline
#     retrieve(question, mode="hybrid")         # phase 2 (to be added)
#
# Implemented so far:
#   - "vector": semantic search in Chroma (embedding similarity)
# ---------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from rag import config
from rag.documents import document_to_chunk
from rag.tracing import observe, update_current_span
from rag.vectorstore import make_embeddings, open_vector_store


@dataclass
class RetrievedChunk:
    """
    One search result: the chunk record (with real lists, ready for
    citations), its relevance score (higher = more relevant) and its
    rank (1 = best).
    """

    chunk: dict[str, Any]
    score: float
    rank: int
    retriever: str

    def summary(self) -> dict[str, Any]:
        """
        Compact view for traces and logs.
        """

        header = self.chunk["text"].split("\n", 1)[0]

        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "chunk_id": self.chunk["chunk_id"],
            "source_url": self.chunk["source_url"],
            "header": header,
        }


class Retriever(Protocol):
    """
    What every retrieval method provides.
    """

    name: str

    def search(self, question: str, k: int) -> list[RetrievedChunk]: ...


# ---------------------------------------------------------
# Retrieval methods
# ---------------------------------------------------------


class VectorRetriever:
    """
    Semantic search: the question is embedded with the same model as
    the chunks, and Chroma returns the chunks with the closest vectors.
    Chroma reports cosine distance (lower = closer); the score here
    is similarity = 1 - distance (higher = more relevant).
    """

    name = "vector"

    def __init__(self) -> None:
        # Question embeddings are not cached (questions are new).
        self.vector_store = open_vector_store(make_embeddings(cached=False))

    def search(self, question: str, k: int) -> list[RetrievedChunk]:
        results = self.vector_store.similarity_search_with_score(question, k=k)

        return [
            RetrievedChunk(chunk=document_to_chunk(document), score=1 - distance, rank=rank, retriever=self.name)
            for rank, (document, distance) in enumerate(results, 1)
        ]


# Name -> retrieval method. Phase 2 adds BM25, hybrid and reranked
# retrieval here.
RETRIEVERS: dict[str, type] = {
    VectorRetriever.name: VectorRetriever,
}


@lru_cache(maxsize=None)
def get_retriever(mode: str) -> Retriever:
    """
    The retriever for a mode, created once and then reused (opening
    Chroma and setting up the embedder takes a moment).
    """

    if mode not in RETRIEVERS:
        raise ValueError(f"Unknown retrieval mode {mode!r}. Available: {', '.join(RETRIEVERS)}")

    return RETRIEVERS[mode]()


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------


@observe(name="retrieve", as_type="retriever", capture_output=False)
def retrieve(question: str, k: int | None = None, mode: str | None = None) -> list[RetrievedChunk]:
    """
    The top k chunks for a question, using the given retrieval mode
    (default: config.RETRIEVAL_MODE). Traced in Langfuse as a
    "retrieve" step with the mode, k and a summary of the results.
    """

    k = k or config.TOP_K
    mode = mode or config.RETRIEVAL_MODE

    results = get_retriever(mode).search(question, k)

    update_current_span(
        metadata={"mode": mode, "k": k},
        output=[result.summary() for result in results],
    )

    return results
