# ---------------------------------------------------------
# pipeline.py
#
# The RAG pipeline: one question in, one answer with citations out.
#
#   answer_question(question)
#     ├─ retrieve()               find the top k chunks
#     └─ generate_cited_answer()  LLM answer citing those chunks
#
# Every entry point uses this one function: the command line
# (scripts/ask_cli.py), the web API (api/main.py) and later the
# evaluation. Each call is one Langfuse trace, tagged with where it
# came from (source="cli", "api", "eval").
# ---------------------------------------------------------

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from rag import config
from rag.generation import Citation, generate_cited_answer
from rag.retrieval import RetrievedChunk, retrieve
from rag.tracing import current_trace_id, observe, trace_attributes, update_current_span


@dataclass
class AnswerResult:
    """
    Everything about one answered question: the answer, its citations,
    all retrieved chunks (for debugging and evaluation) and the
    settings that produced it.
    """

    question: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    retrieval_mode: str
    top_k: int
    llm_model: str
    prompt_version: str
    latency_seconds: float
    trace_id: str | None

    def to_api_response(self) -> dict[str, Any]:
        """
        The JSON shape the React frontend expects (frontend/src/types.ts).
        """

        return {
            "answer": self.answer,
            "citations": [
                {
                    "number": citation.number,
                    "chunkId": citation.chunk_id,
                    "title": citation.title,
                    "breadcrumb": citation.breadcrumb,
                    "headingPath": citation.heading_path,
                    "url": citation.url,
                    "text": citation.text,
                }
                for citation in self.citations
            ],
            "retrieved": [item.summary() for item in self.retrieved_chunks],
            "meta": {
                "retrievalMode": self.retrieval_mode,
                "topK": self.top_k,
                "llmModel": self.llm_model,
                "promptVersion": self.prompt_version,
                "latencySeconds": round(self.latency_seconds, 2),
                "traceId": self.trace_id,
            },
        }


@observe(name="answer_question", capture_output=False)
def run_rag_pipeline(question: str, top_k: int, retrieval_mode: str, prompt_version: str | None) -> AnswerResult:
    started = time.perf_counter()

    retrieved_chunks = retrieve(question, k=top_k, mode=retrieval_mode)
    generated = generate_cited_answer(question, retrieved_chunks, prompt_version)

    result = AnswerResult(
        question=question,
        answer=generated.answer,
        citations=generated.citations,
        retrieved_chunks=retrieved_chunks,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        llm_model=generated.model,
        prompt_version=generated.prompt_version,
        latency_seconds=time.perf_counter() - started,
        trace_id=current_trace_id(),
    )

    update_current_span(
        output={"answer": result.answer, "cited_chunk_ids": [c.chunk_id for c in result.citations]},
        metadata={
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "llm_model": result.llm_model,
            "prompt_version": result.prompt_version,
        },
    )

    return result


def answer_question(
    question: str,
    *,
    top_k: int | None = None,
    retrieval_mode: str | None = None,
    prompt_version: str | None = None,
    source: str = "cli",
) -> AnswerResult:
    """
    Answer one question with citations. `source` labels the trace with
    the caller ("cli", "api", "eval").
    """

    with trace_attributes(trace_name="answer_question", tags=[source]):
        return run_rag_pipeline(
            question,
            top_k or config.TOP_K,
            retrieval_mode or config.RETRIEVAL_MODE,
            prompt_version,
        )
