# ---------------------------------------------------------
# run_eval.py
#
# Evaluation, step 1 of 2: ask the RAG system every question of the
# golden set and save the raw results. Scoring them is a separate step
# (score_eval.py), so a run can be re-scored without asking again.
#
# Every question goes through answer_question(), the same pipeline the
# CLI and the web API use, with the chosen retrieval mode and prompt
# version. Traces are tagged "eval" in Langfuse and grouped under the
# run name as a session.
#
# Output: eval/results/<run_name>/
#   answers.jsonl    one line per question: answer, citations, the
#                    retrieved chunks (rank, score, chunk ID), latency,
#                    LLM tokens and estimated cost, trace ID, error
#   run_config.json  everything needed to reproduce and compare the
#                    run: settings, models, prompt version, checksums of
#                    the corpus and the golden set, git commit, totals
#
# File paths in run_config.json are stored without your home folder
# (repository files relative, e.g. eval/golden_set.jsonl; other files
# as ~/...), so committed results do not reveal the local username.
#
# The golden set is validated first; the run stops if it has problems.
#
# Usage:
#   python scripts/run_eval.py --run-name phase1_vector_v1_run1
#   python scripts/run_eval.py --run-name smoke --questions q01,q10
#   python scripts/run_eval.py --run-name hybrid_v1 --mode hybrid --prompt-version v1
# ---------------------------------------------------------

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make src/ importable when running this file as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import config  # noqa: E402
from rag.documents import load_chunks  # noqa: E402
from rag.generation import get_answer_llm  # noqa: E402
from rag.golden_set import find_golden_set_problems, load_golden_set  # noqa: E402
from rag.pipeline import AnswerResult, answer_question  # noqa: E402
from rag.retrieval import RETRIEVERS, get_retriever  # noqa: E402
from rag.tracing import TRACING_ENABLED, flush  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit_and_dirty() -> tuple[str, bool]:
    """
    The current commit and whether there are uncommitted changes. A run
    from a dirty working tree cannot be reproduced exactly from git.
    """

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], capture_output=True, text=True, cwd=config.REPO_DIR, check=True
        ).stdout.strip()

    try:
        return git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def read_index_manifest() -> dict[str, Any] | None:
    """
    manifest.json written by build_index.py: which chunks file and
    embedding model the Chroma index was built from.
    """

    manifest_path = config.CHROMA_DIR / "manifest.json"

    if not manifest_path.exists():
        return None

    return json.loads(manifest_path.read_text(encoding="utf-8"))


# Removes your Mac's home folder from file paths so saved results don't reveal your username.
def hide_local_paths(value: Any) -> Any:
    """
    Applied to the whole run_config before saving: in every text value,
    the repository path becomes relative (eval/golden_set.jsonl) and the
    home folder becomes ~ (~/Projects/ionos-docs/...). Works through
    nested dicts and lists, e.g. the copied index manifest.
    """

    if isinstance(value, dict):
        return {key: hide_local_paths(item) for key, item in value.items()}

    if isinstance(value, list):
        return [hide_local_paths(item) for item in value]

    if isinstance(value, str):
        return value.replace(f"{config.REPO_DIR}/", "").replace(str(Path.home()), "~")

    return value


def answer_record_from_result(golden_record: dict[str, Any], result: AnswerResult) -> dict[str, Any]:
    """
    The saved form of one answered question. Retrieved chunks are stored
    by ID (the scorer looks their text up in the frozen chunks.jsonl,
    whose checksum is in run_config.json).
    """

    return {
        "id": golden_record["id"],
        "type": golden_record["type"],
        "question": golden_record["question"],
        "answer": result.answer,
        "citations": [
            {"number": citation.number, "chunk_id": citation.chunk_id, "url": citation.url}
            for citation in result.citations
        ],
        "cited_numbers_from_llm": result.cited_numbers_from_llm,
        "retrieved": [
            {
                "rank": item.rank,
                "score": round(item.score, 4),
                "chunk_id": item.chunk["chunk_id"],
                "source_url": item.chunk["source_url"],
            }
            for item in result.retrieved_chunks
        ],
        "latency_seconds": round(result.latency_seconds, 3),
        "llm_token_usage": result.llm_token_usage,
        "llm_cost_usd": result.llm_cost_usd,
        "trace_id": result.trace_id,
        "error": None,
    }


def answer_one_golden_question(golden_record: dict[str, Any], run_settings: dict[str, Any]) -> dict[str, Any]:
    """
    Ask one golden question. An error (e.g. an API timeout) is recorded
    in the result instead of stopping the whole run.
    """

    started = time.perf_counter()

    try:
        result = answer_question(
            golden_record["question"],
            top_k=run_settings["top_k"],
            retrieval_mode=run_settings["retrieval_mode"],
            prompt_version=run_settings["prompt_version"],
            source="eval",
            trace_session_id=run_settings["run_name"],
        )
    except Exception as error:  # noqa: BLE001 (any failure is recorded and the run continues)
        return {
            "id": golden_record["id"],
            "type": golden_record["type"],
            "question": golden_record["question"],
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(error).__name__}: {error}",
        }

    return answer_record_from_result(golden_record, result)


def summarize_answer_records(answer_records: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    """
    Run totals: errors, tokens, cost and latency (not quality; that is
    the scoring step).
    """

    answered = [record for record in answer_records if record["error"] is None]
    latencies = sorted(record["latency_seconds"] for record in answered)
    costs = [record["llm_cost_usd"] for record in answered if record["llm_cost_usd"] is not None]

    def total_tokens(kind: str) -> int:
        return sum(record["llm_token_usage"].get(kind, 0) for record in answered)

    return {
        "questions": len(answer_records),
        "answered": len(answered),
        "errors": len(answer_records) - len(answered),
        "llm_input_tokens": total_tokens("input_tokens"),
        "llm_output_tokens": total_tokens("output_tokens"),
        "llm_cost_usd_total": round(sum(costs), 6) if costs else None,
        "llm_cost_usd_per_answer": round(sum(costs) / len(costs), 6) if costs else None,
        "latency_seconds_mean": round(statistics.mean(latencies), 3) if latencies else None,
        "latency_seconds_p50": round(statistics.median(latencies), 3) if latencies else None,
        "latency_seconds_p95": (
            round(statistics.quantiles(latencies, n=20, method="inclusive")[18], 3) if len(latencies) >= 2 else None
        ),
        "wall_seconds": round(wall_seconds, 1),
    }


# Asks all golden questions, then saves the answers and a record of how the run was done.
def main():

    parser = argparse.ArgumentParser(description="Run the golden set through the RAG pipeline and save the answers")
    parser.add_argument("--run-name", help="folder name under eval/results (default: <mode>_<prompt>_<timestamp>)")
    parser.add_argument("--mode", choices=sorted(RETRIEVERS), default=config.RETRIEVAL_MODE)
    parser.add_argument("--prompt-version", default=config.PROMPT_VERSION)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--questions", help="only these question IDs, comma separated (e.g. q01,q10)")
    parser.add_argument("--workers", type=int, default=4, help="questions answered in parallel")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run with the same name")
    parser.add_argument("--golden-set", type=Path, default=config.GOLDEN_SET_FILE)
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc)
    run_name = args.run_name or f"{args.mode}_{args.prompt_version}_{started_at:%Y%m%d-%H%M}"
    run_dir = config.EVAL_RESULTS_DIR / run_name

    if run_dir.exists() and not args.overwrite:
        sys.exit(f"Run {run_name!r} already exists in {run_dir}. Choose another --run-name or pass --overwrite.")

    # 1. Golden set: validate against the current chunks before spending money.
    golden_records = load_golden_set(args.golden_set)
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in load_chunks(config.CHUNKS_FILE)}
    problems = find_golden_set_problems(golden_records, chunks_by_id)

    if problems:
        sys.exit("Golden set has problems; run scripts/validate_golden_set.py:\n  " + "\n  ".join(problems))

    if args.questions:
        wanted_ids = [question_id.strip() for question_id in args.questions.split(",") if question_id.strip()]
        unknown_ids = set(wanted_ids) - {golden_record["id"] for golden_record in golden_records}
        if unknown_ids:
            sys.exit(f"Unknown question IDs: {', '.join(sorted(unknown_ids))}")
        golden_records = [golden_record for golden_record in golden_records if golden_record["id"] in wanted_ids]

    # 2. Corpus check: the index should be built from the current chunks.
    chunks_sha256 = file_sha256(config.CHUNKS_FILE)
    index_manifest = read_index_manifest()
    index_matches_chunks = bool(index_manifest) and index_manifest.get("chunks_sha256") == chunks_sha256

    if not index_matches_chunks:
        print("WARNING: the Chroma index was not built from the current chunks.jsonl (manifest checksum differs).")

    git_commit, git_dirty = git_commit_and_dirty()
    if git_dirty:
        print("WARNING: uncommitted changes; this run cannot be reproduced exactly from git.")

    run_settings = {
        "run_name": run_name,
        "retrieval_mode": args.mode,
        "top_k": args.top_k,
        "prompt_version": args.prompt_version,
    }

    print(f"Run {run_name}: {len(golden_records)} questions, mode={args.mode}, k={args.top_k}, "
          f"prompt={args.prompt_version}, model={config.LLM_MODEL}, workers={args.workers}\n")

    # 3. Set up the retriever and the LLM once, before the worker threads start.
    get_retriever(args.mode)
    get_answer_llm()

    # 4. Ask every question (a few in parallel).
    run_started = time.perf_counter()
    answers_by_id: dict[str, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(answer_one_golden_question, golden_record, run_settings)
            for golden_record in golden_records
        ]

        for finished_count, future in enumerate(as_completed(futures), 1):
            answer_record = future.result()
            answers_by_id[answer_record["id"]] = answer_record
            status = f"ERROR {answer_record['error']}" if answer_record["error"] else "ok"
            print(f"[{finished_count:>2}/{len(futures)}] {answer_record['id']}  "
                  f"{answer_record['latency_seconds']:5.1f}s  {status}")

    answer_records = [answers_by_id[golden_record["id"]] for golden_record in golden_records]
    totals = summarize_answer_records(answer_records, time.perf_counter() - run_started)

    # 5. Save the answers and the run configuration.
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "answers.jsonl").open("w", encoding="utf-8") as file:
        for answer_record in answer_records:
            file.write(json.dumps(answer_record, ensure_ascii=False) + "\n")

    run_config = {
        "run_name": run_name,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {
            **{key: value for key, value in run_settings.items() if key != "run_name"},
            "llm_model": config.LLM_MODEL,
            "llm_temperature": config.LLM_TEMPERATURE,
            "embedding_model": config.EMBEDDING_MODEL,
            "collection": config.COLLECTION_NAME,
            "workers": args.workers,
        },
        "llm_prices_usd_per_1m_tokens": config.LLM_PRICES_USD_PER_1M_TOKENS.get(config.LLM_MODEL),
        "corpus": {
            "chunks_file": str(config.CHUNKS_FILE),
            "chunks_sha256": chunks_sha256,
            "index_matches_chunks": index_matches_chunks,
            "index_manifest": index_manifest,
        },
        "golden_set": {
            "file": str(args.golden_set),
            "sha256": file_sha256(args.golden_set),
            "question_ids": [golden_record["id"] for golden_record in golden_records],
        },
        "git": {"commit": git_commit, "dirty": git_dirty},
        "langfuse_session_id": run_name if TRACING_ENABLED else None,
        "totals": totals,
    }

    (run_dir / "run_config.json").write_text(
        json.dumps(hide_local_paths(run_config), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    flush()

    print(f"\nSaved to {run_dir}")
    print(f"  answered {totals['answered']}/{totals['questions']}, errors: {totals['errors']}")
    if totals["llm_cost_usd_total"] is not None:
        print(f"  LLM cost: ${totals['llm_cost_usd_total']:.4f} total, ${totals['llm_cost_usd_per_answer']:.5f} per answer")
    if totals["latency_seconds_mean"] is not None:
        print(f"  latency: mean {totals['latency_seconds_mean']}s, p50 {totals['latency_seconds_p50']}s, "
              f"p95 {totals['latency_seconds_p95']}s")
    if TRACING_ENABLED:
        print(f"  Langfuse: Sessions → {run_name}")

    sys.exit(1 if totals["errors"] else 0)


if __name__ == "__main__":
    main()
