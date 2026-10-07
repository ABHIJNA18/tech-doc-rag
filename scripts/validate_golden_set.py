# ---------------------------------------------------------
# validate_golden_set.py
#
# Checks the golden evaluation set (eval/golden_set.jsonl) against the
# current chunks (chunks.jsonl):
#   - every record has all fields with the right types
#   - IDs are unique, types are known
#   - answerable questions have evidence, unanswerable ones have none
#   - every evidence chunk ID exists and contains its quote
#   - source_urls match the pages of the evidence chunks
# and prints the number of questions per type.
#
# Run it after editing the golden set and after re-chunking the docs
# (chunk IDs may change; the quotes show where each fact moved).
# Exits with status 1 when a problem is found.
#
# Usage:
#   python scripts/validate_golden_set.py
#   python scripts/validate_golden_set.py --golden-set path/to/golden_set.jsonl --chunks path/to/chunks.jsonl
# ---------------------------------------------------------

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Make src/ importable when running this file as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import config  # noqa: E402
from rag.documents import load_chunks  # noqa: E402
from rag.golden_set import QUESTION_TYPES, find_golden_set_problems, load_golden_set  # noqa: E402


def main():

    parser = argparse.ArgumentParser(description="Validate the golden evaluation set against chunks.jsonl")
    parser.add_argument("--golden-set", type=Path, default=config.GOLDEN_SET_FILE)
    parser.add_argument("--chunks", type=Path, default=config.CHUNKS_FILE)
    args = parser.parse_args()

    golden_records = load_golden_set(args.golden_set)
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in load_chunks(args.chunks)}

    print(f"Golden set: {args.golden_set} ({len(golden_records)} questions)")
    print(f"Chunks:     {args.chunks} ({len(chunks_by_id)} chunks)\n")

    type_counts = Counter(golden_record.get("type") for golden_record in golden_records)
    for question_type in QUESTION_TYPES:
        print(f"  {question_type:<17} {type_counts.get(question_type, 0)}")

    multi_evidence_count = sum(1 for golden_record in golden_records if len(golden_record.get("evidence", [])) > 1)
    required_terms_count = sum(1 for golden_record in golden_records if golden_record.get("required_terms"))
    print(f"\n  needing several evidence chunks: {multi_evidence_count}")
    print(f"  with required terms:             {required_terms_count}")

    problems = find_golden_set_problems(golden_records, chunks_by_id)

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)

    print("\nOK: no problems found.")


if __name__ == "__main__":
    main()
