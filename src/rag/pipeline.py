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
# (scripts/ask_cli.py), the web API (api/main.py) and the evaluation
# (scripts/run_eval.py). Each call is one Langfuse trace, tagged with
# where it came from (source="cli", "api", "eval"). The result also
# carries the LLM token usage and estimated cost.
# ---------------------------------------------------------

from __future__ import annotations

import time
from dataclasses import dataclass, field
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
    cited_numbers_from_llm: list[int] = field(default_factory=list)
    llm_token_usage: dict[str, int] = field(default_factory=dict)
    llm_cost_usd: float | None = None

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
        cited_numbers_from_llm=generated.cited_numbers_from_llm,
        llm_token_usage=generated.token_usage,
        llm_cost_usd=generated.cost_usd,
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
    trace_session_id: str | None = None,
) -> AnswerResult:
    """
    Answer one question with citations. `source` labels the trace with
    the caller ("cli", "api", "eval"). `trace_session_id` groups traces
    in Langfuse (the evaluation passes its run name, so all answers of
    one run appear together under Sessions).
    """

    with trace_attributes(trace_name="answer_question", tags=[source], session_id=trace_session_id):
        return run_rag_pipeline(
            question,
            top_k or config.TOP_K,
            retrieval_mode or config.RETRIEVAL_MODE,
            prompt_version,
        )
