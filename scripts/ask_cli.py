# ---------------------------------------------------------
# ask_cli.py
#
# Ask the RAG system a question from the command line and print the
# answer with its cited sources. Uses the same answer_question() as
# the web API; traced in Langfuse with the tag "cli".
#
# Usage:
#   python scripts/ask_cli.py "What does a 429 error mean in the AI Model Hub?"
#   python scripts/ask_cli.py "..." --top-k 8 --mode vector
#   python scripts/ask_cli.py "..." --show-retrieved     (also list all retrieved chunks)
#   python scripts/ask_cli.py "..." --show-source-text   (print the cited passages)
# ---------------------------------------------------------

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

# Make src/ importable when running this file as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import config  # noqa: E402
from rag.pipeline import AnswerResult, answer_question  # noqa: E402
from rag.retrieval import RETRIEVERS  # noqa: E402
from rag.tracing import TRACING_ENABLED, flush  # noqa: E402


def print_answer_result(result: AnswerResult, show_retrieved: bool, show_source_text: bool) -> None:

    print("\nAnswer")
    print("-" * 70)
    print(textwrap.fill(result.answer, width=100, replace_whitespace=False))

    print("\nSources")
    print("-" * 70)
    if not result.citations:
        print("(no sources cited)")
    for citation in result.citations:
        location = " > ".join(citation.breadcrumb + citation.heading_path)
        print(f"[{citation.number}] {location}")
        print(f"    {citation.url}")
        if show_source_text:
            print(textwrap.indent(citation.text, "    | "))

    if show_retrieved:
        print("\nRetrieved chunks")
        print("-" * 70)
        cited_ids = {citation.chunk_id for citation in result.citations}
        for item in result.retrieved_chunks:
            marker = "cited" if item.chunk["chunk_id"] in cited_ids else "     "
            print(f"{item.rank}. {item.score:.3f} {marker}  {item.chunk['chunk_id']}")

    print("\n" + "-" * 70)
    print(
        f"mode: {result.retrieval_mode} · top_k: {result.top_k} · model: {result.llm_model} · "
        f"prompt: {result.prompt_version} · {result.latency_seconds:.1f}s"
    )


def main():

    parser = argparse.ArgumentParser(description="Ask the RAG system a question")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--mode", choices=sorted(RETRIEVERS), default=config.RETRIEVAL_MODE)
    parser.add_argument("--prompt-version", default=config.PROMPT_VERSION)
    parser.add_argument("--show-retrieved", action="store_true", help="list all retrieved chunks")
    parser.add_argument("--show-source-text", action="store_true", help="print the cited passages")
    args = parser.parse_args()

    print(f"Question: {args.question}")

    result = answer_question(
        args.question,
        top_k=args.top_k,
        retrieval_mode=args.mode,
        prompt_version=args.prompt_version,
        source="cli",
    )

    print_answer_result(result, args.show_retrieved, args.show_source_text)

    flush()
    if TRACING_ENABLED:
        print(f"Langfuse trace: {result.trace_id}")


if __name__ == "__main__":
    main()
