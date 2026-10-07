# ---------------------------------------------------------
# golden_set.py
#
# The golden evaluation set (eval/golden_set.jsonl): loading it and
# checking that it is consistent with the indexed chunks.
#
# One record per question:
#   id, type, question, expected_answer,
#   evidence:       [{quote, chunk_ids}], one item per fact the answer
#                   needs; any chunk listed in an item counts as
#                   finding that fact
#   source_urls:    pages of all evidence chunks
#   required_terms: exact identifiers the answer must contain
#   answerable:     false for questions the docs cannot answer
#                   (these have no evidence)
#   notes
#
# Only the standard library is used here, so the scoring step (which
# runs in its own virtual environment with RAGAS) can import this file
# too.
# ---------------------------------------------------------

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

QUESTION_TYPES = (
    "api_fact",
    "exact_identifier",
    "how_to",
    "faq",
    "paraphrased",
    "multi_chunk",
    "definition",
    "unanswerable",
)

GOLDEN_RECORD_FIELDS: dict[str, type] = {
    "id": str,
    "type": str,
    "question": str,
    "expected_answer": str,
    "evidence": list,
    "source_urls": list,
    "required_terms": list,
    "answerable": bool,
    "notes": str,
}


def load_golden_set(path: Path) -> list[dict[str, Any]]:
    """
    Read golden_set.jsonl: one question record per line.
    """

    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def evidence_chunk_ids(golden_record: dict[str, Any]) -> set[str]:
    """
    All chunk IDs that hold evidence for this question (any evidence item).
    """

    return {chunk_id for item in golden_record["evidence"] for chunk_id in item["chunk_ids"]}


def find_golden_record_problems(
    golden_record: dict[str, Any], chunks_by_id: dict[str, dict[str, Any]]
) -> list[str]:
    """
    Everything wrong with one record, as readable messages (empty list
    = record is fine).
    """

    record_id = golden_record.get("id", "<no id>")
    problems = []

    for field_name, field_type in GOLDEN_RECORD_FIELDS.items():
        if not isinstance(golden_record.get(field_name), field_type):
            problems.append(f"{record_id}: field {field_name!r} missing or not a {field_type.__name__}")

    if problems:
        return problems

    if golden_record["type"] not in QUESTION_TYPES:
        problems.append(f"{record_id}: unknown type {golden_record['type']!r}")

    if not golden_record["question"].strip() or not golden_record["expected_answer"].strip():
        problems.append(f"{record_id}: empty question or expected_answer")

    if golden_record["answerable"] != bool(golden_record["evidence"]):
        problems.append(f"{record_id}: answerable questions need evidence, unanswerable ones must have none")

    if (golden_record["type"] == "unanswerable") == golden_record["answerable"]:
        problems.append(
            f"{record_id}: type {golden_record['type']!r} does not match answerable={golden_record['answerable']}"
            " (only type 'unanswerable' has answerable=false)"
        )

    expected_source_urls = []

    for item_number, evidence_item in enumerate(golden_record["evidence"], 1):
        quote = evidence_item.get("quote", "")
        chunk_ids = evidence_item.get("chunk_ids", [])

        if not quote.strip() or not chunk_ids:
            problems.append(f"{record_id}: evidence item {item_number} needs a quote and at least one chunk ID")
            continue

        for chunk_id in chunk_ids:
            chunk = chunks_by_id.get(chunk_id)

            if chunk is None:
                problems.append(f"{record_id}: chunk {chunk_id} does not exist in chunks.jsonl")
                continue

            # Case-insensitive, so the quote survives small casing changes.
            if quote.lower() not in chunk["text"].lower():
                problems.append(f"{record_id}: quote {quote!r} not found in chunk {chunk_id}")

            if chunk["source_url"] not in expected_source_urls:
                expected_source_urls.append(chunk["source_url"])

    if set(golden_record["source_urls"]) != set(expected_source_urls):
        problems.append(f"{record_id}: source_urls do not match the pages of the evidence chunks")

    return problems


def find_golden_set_problems(
    golden_records: list[dict[str, Any]], chunks_by_id: dict[str, dict[str, Any]]
) -> list[str]:
    """
    Problems across the whole set: every record's problems plus
    duplicate IDs.
    """

    problems = []

    for golden_record in golden_records:
        problems.extend(find_golden_record_problems(golden_record, chunks_by_id))

    id_counts = Counter(golden_record.get("id") for golden_record in golden_records)
    problems.extend(f"duplicate id {record_id}" for record_id, count in id_counts.items() if count > 1)

    return problems
