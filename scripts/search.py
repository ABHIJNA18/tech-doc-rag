# ---------------------------------------------------------
# search.py
#
# Try retrieval from the command line: prints the top chunks for a
# question with their scores. Each search is traced in Langfuse
# (when the keys are set), so you can open it in the dashboard.
#
# Usage:
#   python scripts/search.py "What does a 429 error mean in AI Model Hub?"
#   python scripts/search.py "max replicas In-Memory DB" --k 3
#   python scripts/search.py "..." --mode vector
#   python scripts/search.py "..." --full          (print full chunk texts)
# ---------------------------------------------------------

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable when running this file as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import config  # noqa: E402
from rag.retrieval import RETRIEVERS, retrieve  # noqa: E402
from rag.tracing import TRACING_ENABLED, flush, observe  # noqa: E402


@observe(name="search-cli", capture_output=False)
def search(question: str, k: int, mode: str):
    """
    One traced search. The Langfuse trace shows this step with the
    "retrieve" step inside it.
    """

    return retrieve(question, k=k, mode=mode)


def main():

    parser = argparse.ArgumentParser(description="Search the index from the command line")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--mode", choices=sorted(RETRIEVERS), default=config.RETRIEVAL_MODE)
    parser.add_argument("--full", action="store_true", help="print the full chunk texts")
    args = parser.parse_args()

    results = search(args.question, args.k, args.mode)

    print(f"Question: {args.question}")
    print(f"Mode: {args.mode}   k: {args.k}\n")

    for result in results:
        chunk = result.chunk
        header = chunk["text"].split("\n", 1)[0]
        print(f"{result.rank}. score {result.score:.3f}  {header}")
        print(f"   {chunk['source_url']}")
        print(f"   {chunk['chunk_id']}")

        if args.full:
            print("\n" + chunk["text"] + "\n")

    flush()
    print(f"\nLangfuse tracing: {'on (see your Langfuse project → Tracing)' if TRACING_ENABLED else 'off (no keys)'}")


if __name__ == "__main__":
    main()
