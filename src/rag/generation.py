# ---------------------------------------------------------
# generation.py
#
# Generating an answer with citations from retrieved chunks.
#
#   1. format_sources_for_prompt(): number the retrieved chunks
#      [1]..[k] and put them, as text, into the prompt
#   2. generate_cited_answer(): call the chat LLM with the versioned
#      prompt; structured output returns the answer text (with [n]
#      markers) and the source numbers it cited
#   3. build_citations(): map each cited [n] back to its chunk (URL,
#      breadcrumb, section, exact text) for display
#
# The LLM decides which sources it used; our code turns the numbers
# into real sources. Checking that citations are honest is phase 2
# (citation enforcement).
# ---------------------------------------------------------

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from rag import config
from rag.prompts import load_prompt_template
from rag.retrieval import RetrievedChunk
from rag.tracing import langchain_callbacks, observe

ANSWER_PROMPT_NAME = "answer_with_citations"

CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


class LLMCitedAnswer(BaseModel):
    """
    The structured output the LLM must return.
    """

    answer: str = Field(description="The answer, with [n] citation markers after the sentences they support.")
    cited_sources: list[int] = Field(description="The numbers of all sources cited in the answer.")


@dataclass
class Citation:
    """
    One source cited in the answer, ready to display.
    """

    number: int
    chunk_id: str
    title: str
    breadcrumb: list[str]
    heading_path: list[str]
    url: str
    text: str


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[Citation]
    model: str
    prompt_version: str
    cited_numbers_from_llm: list[int] = field(default_factory=list)


@lru_cache(maxsize=None)
def get_answer_llm() -> Any:
    """
    The chat model, set up once, constrained to return LLMCitedAnswer.
    """

    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)

    return llm.with_structured_output(LLMCitedAnswer)


def format_sources_for_prompt(retrieved_chunks: list[RetrievedChunk]) -> str:
    """
    The retrieved chunks as numbered sources. The chunk text already
    starts with its header (breadcrumb > page > section).
    """

    return "\n\n".join(
        f"[{number}] (source: {item.chunk['source_url']})\n{item.chunk['text']}"
        for number, item in enumerate(retrieved_chunks, 1)
    )


def chunk_body(chunk: dict[str, Any]) -> str:
    """
    The chunk text without its header line, for display (the header
    is shown separately as breadcrumb > section).
    """

    header, _, body = chunk["text"].partition("\n\n")

    return body or header


def build_citations(answer_text: str, cited_numbers: list[int], retrieved_chunks: list[RetrievedChunk]) -> list[Citation]:
    """
    Map cited source numbers back to their chunks. Numbers come from
    the [n] markers in the answer (in order of first appearance) plus
    any extra numbers the LLM listed; numbers outside 1..k are ignored.
    """

    numbers_in_text = [int(match) for match in CITATION_MARKER_RE.findall(answer_text)]

    ordered_numbers = []
    for number in numbers_in_text + list(cited_numbers):
        if 1 <= number <= len(retrieved_chunks) and number not in ordered_numbers:
            ordered_numbers.append(number)

    citations = []

    for number in ordered_numbers:
        chunk = retrieved_chunks[number - 1].chunk
        citations.append(
            Citation(
                number=number,
                chunk_id=chunk["chunk_id"],
                title=chunk["document_title"],
                breadcrumb=chunk["breadcrumb"],
                heading_path=chunk["heading_path"],
                url=chunk["source_url"],
                text=chunk_body(chunk),
            )
        )

    return citations


@observe(name="generate_cited_answer", capture_input=False, capture_output=False)
def generate_cited_answer(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    prompt_version: str | None = None,
) -> GeneratedAnswer:
    """
    Ask the LLM to answer the question from the retrieved chunks. The
    LLM call itself is traced by the Langfuse LangChain callback (exact
    prompt, model, tokens, cost).
    """

    prompt_template, prompt_version = load_prompt_template(ANSWER_PROMPT_NAME, prompt_version)

    prompt_messages = prompt_template.invoke(
        {"sources": format_sources_for_prompt(retrieved_chunks), "question": question}
    )

    llm_output: LLMCitedAnswer = get_answer_llm().invoke(
        prompt_messages,
        config={"callbacks": langchain_callbacks(), "run_name": "llm_answer"},
    )

    return GeneratedAnswer(
        answer=llm_output.answer,
        citations=build_citations(llm_output.answer, llm_output.cited_sources, retrieved_chunks),
        model=config.LLM_MODEL,
        prompt_version=prompt_version,
        cited_numbers_from_llm=llm_output.cited_sources,
    )
