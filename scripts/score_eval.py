# ---------------------------------------------------------
# score_eval.py
#
# Evaluation, step 2 of 2: grade the answers of one run (saved by
# run_eval.py) against the golden set.
#
# Code checks (free, always the same result):
#   - retrieval: rank of the first evidence chunk, hit@1/3/5,
#     reciprocal rank (for MRR), evidence recall (for questions that
#     need several facts: how many of them were retrieved)
#   - required terms: exact identifiers the answer must contain
#   - citations: does the answer cite a chunk that holds the evidence,
#     and does it use source numbers that do not exist
#
# LLM judge checks (RAGAS, judge model config.EVAL_JUDGE_MODEL, by
# default Claude Sonnet 5.5 through the Anthropic SDK, ClaudeJudgeLLM):
#   - faithfulness: share of the answer's claims supported by the
#     retrieved chunks (made-up facts lower it)
#   - factual correctness (recall mode): share of the golden answer's
#     facts that the answer contains
#   - context recall: share of the golden answer's facts that the
#     retrieved chunks contain
#   - declines: does the answer say the docs do not cover it (good for
#     unanswerable questions, a false refusal for answerable ones)
#
# Runs in its own virtual environment (.venv-eval, see
# requirements-eval.txt), because RAGAS needs older library versions
# than the RAG app.
#
# Output in eval/results/<run_name>/:
#   scores.jsonl   one line per question with every check
#   summary.json   averages overall and per question type, cost, latency
#   summary.md     the same as readable tables, plus the failures
# With --questions (a subset), the files are named scores_subset.* so a
# trial does not look like a full result.
#
# A run can be re-graded after the answer key (expected answers, notes)
# was improved, as long as its questions are unchanged; summary.json
# records which answer key version did the grading.
#
# Usage:
#   .venv-eval/bin/python scripts/score_eval.py phase1_vector_v1_run1
#   .venv-eval/bin/python scripts/score_eval.py phase1_vector_v1_run1 --questions q01,q10,q13,q19,q34
#   .venv-eval/bin/python scripts/score_eval.py phase1_vector_v1_run1 --no-judge   (code checks only, free)
# ---------------------------------------------------------

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# RAGAS sends anonymous usage statistics unless this is set.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

# Make src/ importable when running this file as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import config  # noqa: E402  (also loads the API keys from .env)
from rag.golden_set import QUESTION_TYPES, evidence_chunk_ids, load_golden_set  # noqa: E402
from rag.llm_costs import estimate_llm_cost_usd  # noqa: E402

import ragas  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.llms.base import InstructorBaseRagasLLM  # noqa: E402
from ragas.metrics import DiscreteMetric  # noqa: E402
from ragas.metrics.collections import ContextRecall, FactualCorrectness, Faithfulness  # noqa: E402

CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")

DECLINES_PROMPT = """You are grading a documentation assistant's response to a user question.

Question: {question}

Response: {response}

Does the response DECLINE to answer? Answer "yes" if its main message is that the provided
documentation does not contain the requested information, or that the question is unclear
and needs clarification. It still counts as declining if it adds related context or points to
where the information might be found. Answer "no" if the response presents an answer to the
question, including an answer based on a guessed interpretation of an unclear question."""

JUDGE_METRIC_NAMES = ("faithfulness", "factual_correctness", "context_recall", "declines")


# ---------------------------------------------------------
# Loading
# ---------------------------------------------------------


# Calculates a fingerprint of a file, to check the docs have not changed since the run.
def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Finds the last git commit that changed the answer key, to record which version graded a run.
def golden_set_last_commit(golden_set_file: Path) -> str:
    """
    The short hash of the last commit touching the file, plus "-dirty"
    when the file has uncommitted changes ("unknown" without git).
    """

    import subprocess

    # Runs one git command in the project folder and returns its output.
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], capture_output=True, text=True, cwd=config.REPO_DIR, check=True
        ).stdout.strip()

    try:
        commit = git("log", "-1", "--format=%h", "--", str(golden_set_file))
        dirty = bool(git("status", "--porcelain", "--", str(golden_set_file)))
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

    return f"{commit}-dirty" if dirty else commit


# Reads one saved file with one JSON record per line.
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


# ---------------------------------------------------------
# Code checks
# ---------------------------------------------------------


# Removes Markdown formatting (backticks, bold) so `allkeys-lru` and allkeys-lru count the same.
def plain_answer_text(answer: str) -> str:
    return re.sub(r"[`*]", "", answer).lower()


# Checks search, must-have words and citations for one question with plain code.
def code_checks(golden_record: dict[str, Any], answer_record: dict[str, Any]) -> dict[str, Any]:
    """
    Retrieval metrics are only computed for answerable questions (the
    others have no evidence). A cited chunk "holds evidence" when it is
    one of the evidence chunks of the question.
    """

    retrieved_ids = [item["chunk_id"] for item in answer_record["retrieved"]]
    retrieved_count = len(retrieved_ids)
    answer = answer_record["answer"]

    checks: dict[str, Any] = {}

    if golden_record["answerable"]:
        evidence_ids = evidence_chunk_ids(golden_record)
        first_evidence_rank = next(
            (rank for rank, chunk_id in enumerate(retrieved_ids, 1) if chunk_id in evidence_ids), None
        )
        evidence_items_found = sum(
            1 for item in golden_record["evidence"] if set(item["chunk_ids"]) & set(retrieved_ids)
        )
        cited_ids = {citation["chunk_id"] for citation in answer_record["citations"]}

        checks.update(
            {
                "first_evidence_rank": first_evidence_rank,
                "hit_at_1": first_evidence_rank is not None and first_evidence_rank <= 1,
                "hit_at_3": first_evidence_rank is not None and first_evidence_rank <= 3,
                "hit_at_5": first_evidence_rank is not None and first_evidence_rank <= 5,
                "reciprocal_rank": 1 / first_evidence_rank if first_evidence_rank else 0.0,
                "evidence_items_found": evidence_items_found,
                "evidence_items_total": len(golden_record["evidence"]),
                "evidence_recall": evidence_items_found / len(golden_record["evidence"]),
                "cites_evidence_chunk": bool(cited_ids & evidence_ids),
            }
        )

    plain_answer = plain_answer_text(answer)
    missing_terms = [term for term in golden_record["required_terms"] if term.lower() not in plain_answer]

    citation_numbers = [int(number) for number in CITATION_MARKER_RE.findall(answer)]

    checks.update(
        {
            "required_terms_missing": missing_terms,
            "required_terms_ok": (not missing_terms) if golden_record["required_terms"] else None,
            "has_citations": bool(answer_record["citations"]),
            "invalid_citation_numbers": sorted(
                {number for number in citation_numbers if not 1 <= number <= retrieved_count}
            ),
        }
    )

    return checks


# ---------------------------------------------------------
# LLM judge checks (RAGAS)
# ---------------------------------------------------------


class JudgeUsageCounter:
    """
    Adds up the tokens and estimated cost of every judge call (RAGAS
    does not report them), and which model answered each call (a
    Claude refusal fallback can hand a call to another model).
    """

    # Starts all counts at zero.
    def __init__(self) -> None:
        self.calls = 0
        self.token_usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "reasoning_tokens": 0}
        self.cost_usd: float | None = 0.0
        self.calls_per_model: Counter[str] = Counter()

    # Adds one judge call's tokens and cost to the totals.
    def add_call(self, model: str, token_usage: dict[str, int]) -> None:
        self.calls += 1
        self.calls_per_model[model] += 1

        for kind, count in token_usage.items():
            self.token_usage[kind] += count

        call_cost = estimate_llm_cost_usd(model, token_usage)
        # One call with an unknown price makes the total unknown.
        self.cost_usd = None if call_cost is None or self.cost_usd is None else self.cost_usd + call_cost

    # Reads the token counts from one OpenAI reply (used when the judge is an OpenAI model).
    async def record_openai_response(self, response: Any) -> None:
        if response.status_code != 200 or not response.request.url.path.endswith("/chat/completions"):
            return

        await response.aread()
        body = response.json()
        usage = body.get("usage") or {}

        self.add_call(
            body.get("model", config.EVAL_JUDGE_MODEL),
            {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cached_input_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
            },
        )


class ClaudeJudgeLLM(InstructorBaseRagasLLM):
    """
    Lets RAGAS use a Claude model as the judge, through the official
    Anthropic SDK. RAGAS asks every question as agenerate(prompt,
    answer_shape); this sends the prompt to Claude with structured
    outputs, so the reply is guaranteed to match answer_shape.

    Why not RAGAS's built-in Claude support: it sends temperature/top_p
    and forces a tool call to get JSON back, and Claude Sonnet 5.5
    rejects both. Thinking stays on (it cannot be switched off on this
    model) at a low effort level, which suits short grading decisions.
    A server-side fallback lets another Claude model finish a call
    that Claude Sonnet 5.5 declines; which model answered each call is
    counted, so a fallback is visible in summary.json.
    """

    # Sets up the Claude client (the key comes from ANTHROPIC_API_KEY in .env).
    def __init__(self, model: str, effort: str, usage_counter: JudgeUsageCounter) -> None:
        import anthropic

        self.client = anthropic.AsyncAnthropic()
        self.model = model
        self.effort = effort
        self.usage_counter = usage_counter

    # Asks Claude one grading question and returns its answer in the shape RAGAS expects.
    async def agenerate(self, prompt: str, response_model: Any) -> Any:
        response = await self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
            output_format=response_model,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

        usage = response.usage
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.usage_counter.add_call(
            response.model,
            {
                "input_tokens": usage.input_tokens + cache_read_tokens + cache_write_tokens,
                "output_tokens": usage.output_tokens,
                "cached_input_tokens": cache_read_tokens,
                "reasoning_tokens": 0,  # Claude bills thinking as output tokens, already counted above
            },
        )

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            raise RuntimeError(f"Claude declined to grade (category: {category})")

        if response.parsed_output is None:
            raise ValueError(f"Claude returned no structured answer (stop_reason: {response.stop_reason})")

        return response.parsed_output

    # RAGAS requires a non-async version too; this grader only uses agenerate().
    def generate(self, prompt: str, response_model: Any) -> Any:
        raise NotImplementedError("score_eval.py grades asynchronously; use agenerate()")


# Sets up the judge LLM (Claude or OpenAI, from config.EVAL_JUDGE_MODEL) and the four RAGAS checks.
def make_judge_metrics(usage_counter: JudgeUsageCounter) -> tuple[Any, dict[str, Any]]:
    """
    Claude models go through ClaudeJudgeLLM. OpenAI models go through
    RAGAS's own llm_factory, with two fixes for gpt-5 models.
    """

    if config.EVAL_JUDGE_MODEL.startswith("claude-"):
        judge_llm = ClaudeJudgeLLM(config.EVAL_JUDGE_MODEL, config.EVAL_JUDGE_EFFORT, usage_counter)
    else:
        import httpx
        from openai import AsyncOpenAI

        http_client = httpx.AsyncClient(timeout=120, event_hooks={"response": [usage_counter.record_openai_response]})
        # max_tokens is raised from RAGAS's default 1024, which is too low
        # for structured output from gpt-5 models.
        judge_llm = llm_factory(config.EVAL_JUDGE_MODEL, client=AsyncOpenAI(http_client=http_client), max_tokens=4096)

        # RAGAS 0.4.3 renames max_tokens to max_completion_tokens for gpt-5
        # models, but does not recognise names like "gpt-5.4" (it reads the
        # version as a whole number), and the API rejects max_tokens.
        if "max_tokens" in judge_llm.model_args:
            judge_llm.model_args["max_completion_tokens"] = judge_llm.model_args.pop("max_tokens")

    judge_metrics = {
        "faithfulness": Faithfulness(llm=judge_llm),
        "factual_correctness": FactualCorrectness(llm=judge_llm, mode="recall"),
        "context_recall": ContextRecall(llm=judge_llm),
        "declines": DiscreteMetric(name="declines", allowed_values=["yes", "no"], prompt=DECLINES_PROMPT),
    }

    return judge_llm, judge_metrics


# Turns a "not a number" score into an empty value, so it can be saved as JSON.
def clean_score(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None

    return value


# Asks the judge LLM all checks for all questions, a few at a time.
async def judge_checks_for_questions(
    scoring_items: list[tuple[dict[str, Any], dict[str, Any], list[str]]],
    concurrency: int,
    usage_counter: JudgeUsageCounter,
) -> dict[str, dict[str, Any]]:
    """
    scoring_items: (golden record, answer record, retrieved chunk texts).
    Correctness and context recall need a golden answer, so they only
    run for answerable questions. A failed judge call is recorded as an
    error for that check; the other checks still run.
    """

    judge_llm, judge_metrics = make_judge_metrics(usage_counter)
    semaphore = asyncio.Semaphore(concurrency)

    # Runs one judge check for one question and returns its score (or the error).
    async def run_one_check(question_id: str, metric_name: str, score_call) -> tuple[str, str, dict[str, Any]]:
        async with semaphore:
            try:
                result = await score_call()
            except Exception as error:  # noqa: BLE001 (recorded per check)
                return question_id, metric_name, {"value": None, "error": f"{type(error).__name__}: {error}"}

        return question_id, metric_name, {"value": clean_score(result.value), "reason": getattr(result, "reason", None)}

    tasks = []

    for golden_record, answer_record, retrieved_texts in scoring_items:
        question_id = golden_record["id"]
        question = golden_record["question"]
        answer = answer_record["answer"]

        tasks.append(run_one_check(question_id, "faithfulness", lambda q=question, a=answer, t=retrieved_texts:
            judge_metrics["faithfulness"].ascore(user_input=q, response=a, retrieved_contexts=t)))

        tasks.append(run_one_check(question_id, "declines", lambda q=question, a=answer:
            judge_metrics["declines"].ascore(llm=judge_llm, question=q, response=a)))

        if golden_record["answerable"]:
            reference = golden_record["expected_answer"]

            tasks.append(run_one_check(question_id, "factual_correctness", lambda a=answer, r=reference:
                judge_metrics["factual_correctness"].ascore(response=a, reference=r)))

            tasks.append(run_one_check(question_id, "context_recall", lambda q=question, t=retrieved_texts, r=reference:
                judge_metrics["context_recall"].ascore(user_input=q, retrieved_contexts=t, reference=r)))

    judge_results: dict[str, dict[str, Any]] = {}
    finished = 0

    for next_finished in asyncio.as_completed(tasks):
        question_id, metric_name, outcome = await next_finished
        judge_results.setdefault(question_id, {})[metric_name] = outcome
        finished += 1
        if finished % 20 == 0 or finished == len(tasks):
            print(f"  judge checks done: {finished}/{len(tasks)}")

    return judge_results


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------


# Averages a list of numbers, ignoring empty values (None if nothing is left).
def mean_or_none(values: list[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None]

    return round(statistics.mean(numbers), 3) if numbers else None


# Share of True values in a list, ignoring empty values (None if nothing is left).
def rate_or_none(values: list[Any]) -> float | None:
    booleans = [bool(value) for value in values if value is not None]

    return round(sum(booleans) / len(booleans), 3) if booleans else None


# Calculates all average scores for a group of graded questions.
def summarize_score_records(score_records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Retrieval, correctness and context recall are averaged over
    answerable questions; refusal accuracy over unanswerable ones; the
    false refusal rate over answerable ones.
    """

    graded = [record for record in score_records if record.get("error") is None]
    answerable = [record for record in graded if record["answerable"]]
    unanswerable = [record for record in graded if not record["answerable"]]
    multi_evidence = [record for record in answerable if record["code"]["evidence_items_total"] > 1]

    # Value of one judge check for one question (None if not run or failed).
    def judge_value(record: dict[str, Any], metric_name: str) -> Any:
        return (record.get("judge") or {}).get(metric_name, {}).get("value")

    # True/False for "the answer declines", from the judge's yes/no (None if not judged).
    def declined(record: dict[str, Any]) -> bool | None:
        value = judge_value(record, "declines")
        return None if value is None else value == "yes"

    return {
        "questions": len(score_records),
        "graded": len(graded),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "retrieval": {
            "hit_at_1": rate_or_none([record["code"]["hit_at_1"] for record in answerable]),
            "hit_at_3": rate_or_none([record["code"]["hit_at_3"] for record in answerable]),
            "hit_at_5": rate_or_none([record["code"]["hit_at_5"] for record in answerable]),
            "mrr": mean_or_none([record["code"]["reciprocal_rank"] for record in answerable]),
            "evidence_recall": mean_or_none([record["code"]["evidence_recall"] for record in answerable]),
            "all_evidence_found_multi": rate_or_none(
                [record["code"]["evidence_recall"] == 1 for record in multi_evidence]
            ),
        },
        "answer": {
            "required_terms_ok": rate_or_none([record["code"]["required_terms_ok"] for record in graded]),
            "cites_evidence_chunk": rate_or_none([record["code"]["cites_evidence_chunk"] for record in answerable]),
            "invalid_citation_answers": sum(1 for record in graded if record["code"]["invalid_citation_numbers"]),
            "faithfulness": mean_or_none([judge_value(record, "faithfulness") for record in answerable]),
            "factual_correctness": mean_or_none([judge_value(record, "factual_correctness") for record in answerable]),
            "context_recall": mean_or_none([judge_value(record, "context_recall") for record in answerable]),
            "refusal_accuracy": rate_or_none([declined(record) for record in unanswerable]),
            "false_refusal_rate": rate_or_none([declined(record) for record in answerable]),
        },
    }


# Lists the questions that lost marks, with the reason, for the report.
def find_failures(score_records: list[dict[str, Any]]) -> list[str]:
    failures = []

    for record in score_records:
        question_id = f"{record['id']} ({record['type']})"
        judge = record.get("judge") or {}

        if record.get("error"):
            failures.append(f"{question_id}: run error: {record['error']}")
            continue

        code = record["code"]
        problems = []

        if record["answerable"]:
            if not code["hit_at_5"]:
                problems.append("evidence not retrieved in top 5")
            elif not code["hit_at_1"]:
                problems.append(f"evidence at rank {code['first_evidence_rank']}")
            if code["evidence_recall"] < 1:
                problems.append(f"evidence {code['evidence_items_found']}/{code['evidence_items_total']}")
            if judge.get("declines", {}).get("value") == "yes":
                problems.append("false refusal")
            correctness = judge.get("factual_correctness", {}).get("value")
            if correctness is not None and correctness < 1:
                problems.append(f"correctness {correctness:.2f}")
        elif judge.get("declines", {}).get("value") == "no":
            problems.append("answered instead of declining")

        faithfulness = judge.get("faithfulness", {}).get("value")
        if faithfulness is not None and faithfulness < 1:
            problems.append(f"faithfulness {faithfulness:.2f}")
        if code["required_terms_missing"]:
            problems.append(f"missing terms {code['required_terms_missing']}")
        if code["invalid_citation_numbers"]:
            problems.append(f"invalid citations {code['invalid_citation_numbers']}")

        failed_checks = [name for name, outcome in judge.items() if outcome.get("error")]
        if failed_checks:
            problems.append(f"judge errors: {failed_checks}")

        if problems:
            failures.append(f"{question_id}: " + "; ".join(problems))

    return failures


# Formats a score for the report: percentages for rates, 3 decimals for averages.
def format_score(value: Any, as_percent: bool) -> str:
    if value is None:
        return "–"

    return f"{value * 100:.0f}%" if as_percent else f"{value:.3f}"


# Writes the readable report (summary.md) with the overall table, the per-type table and the failures.
def summary_markdown(run_name: str, summary: dict[str, Any], failures: list[str]) -> str:
    overall = summary["overall"]

    rows = [
        ("hit@1", overall["retrieval"]["hit_at_1"], True),
        ("hit@3", overall["retrieval"]["hit_at_3"], True),
        ("hit@5", overall["retrieval"]["hit_at_5"], True),
        ("MRR", overall["retrieval"]["mrr"], False),
        ("Evidence recall", overall["retrieval"]["evidence_recall"], False),
        ("All evidence found (multi-fact questions)", overall["retrieval"]["all_evidence_found_multi"], True),
        ("Faithfulness (RAGAS)", overall["answer"]["faithfulness"], False),
        ("Factual correctness, recall (RAGAS)", overall["answer"]["factual_correctness"], False),
        ("Context recall (RAGAS)", overall["answer"]["context_recall"], False),
        ("Required terms present", overall["answer"]["required_terms_ok"], True),
        ("Cites an evidence chunk", overall["answer"]["cites_evidence_chunk"], True),
        ("Refusal accuracy (unanswerable)", overall["answer"]["refusal_accuracy"], True),
        ("False refusal rate (answerable)", overall["answer"]["false_refusal_rate"], True),
    ]

    lines = [
        f"# Evaluation summary: {run_name}",
        "",
        f"Graded {overall['graded']}/{overall['questions']} questions "
        f"({overall['answerable']} answerable, {overall['unanswerable']} unanswerable). "
        f"Judge: {summary['judge']['model'] or 'none (code checks only)'}.",
        "",
        "| Metric | Score |",
        "| --- | --- |",
        *[f"| {name} | {format_score(value, as_percent)} |" for name, value, as_percent in rows],
        f"| Answers with invalid citation numbers | {overall['answer']['invalid_citation_answers']} |",
        "",
        "## Cost and latency",
        "",
        f"- Answer LLM: ${summary['run_totals'].get('llm_cost_usd_per_answer') or 0:.5f} per answer, "
        f"latency p50 {summary['run_totals'].get('latency_seconds_p50')}s, p95 {summary['run_totals'].get('latency_seconds_p95')}s",
        (
            f"- Judge: {summary['judge']['model']}"
            + (f" (effort {summary['judge']['effort']})" if summary["judge"]["effort"] else "")
            + f", ${summary['judge']['cost_usd'] or 0:.4f} for {summary['judge']['calls']} calls "
            f"({summary['judge']['seconds']}s); calls per model: {summary['judge']['calls_per_model']}"
            if summary["judge"]["model"]
            else "- Judge: not run (code checks only)"
        ),
        "",
        "## By question type",
        "",
        "| Type | n | hit@1 | hit@5 | MRR | Faithfulness | Correctness | Declined |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for question_type, type_summary in summary["by_type"].items():
        retrieval = type_summary["retrieval"]
        answer = type_summary["answer"]
        declined_rate = answer["refusal_accuracy"] if question_type == "unanswerable" else answer["false_refusal_rate"]
        lines.append(
            f"| {question_type} | {type_summary['graded']} | {format_score(retrieval['hit_at_1'], True)} | "
            f"{format_score(retrieval['hit_at_5'], True)} | {format_score(retrieval['mrr'], False)} | "
            f"{format_score(answer['faithfulness'], False)} | {format_score(answer['factual_correctness'], False)} | "
            f"{format_score(declined_rate, True)} |"
        )

    lines += [
        "",
        "Declined: share of answers that said the docs do not cover it "
        "(good for unanswerable questions, a false refusal otherwise).",
        "",
        "## Questions that lost marks",
        "",
        *[f"- {failure}" for failure in failures],
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


# Grades one run: code checks, judge checks, then saves the scores and the reports.
def main():

    parser = argparse.ArgumentParser(description="Grade the answers of one evaluation run")
    parser.add_argument("run_name", help="folder name under eval/results")
    parser.add_argument("--questions", help="only these question IDs, comma separated (writes scores_subset.*)")
    parser.add_argument("--no-judge", action="store_true", help="code checks only (free, no LLM calls)")
    parser.add_argument("--judge-concurrency", type=int, default=8, help="judge calls running at the same time")
    args = parser.parse_args()

    run_dir = config.EVAL_RESULTS_DIR / args.run_name
    if not (run_dir / "answers.jsonl").exists():
        sys.exit(f"No answers found in {run_dir}. Run scripts/run_eval.py first.")

    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    # 1. The docs and the golden set must be the ones the run used.
    chunks_file = Path(run_config["corpus"]["chunks_file"]).expanduser()
    if file_sha256(chunks_file) != run_config["corpus"]["chunks_sha256"]:
        sys.exit("chunks.jsonl changed since this run; scores would be wrong. Re-run run_eval.py.")

    golden_set_file = config.REPO_DIR / run_config["golden_set"]["file"]
    golden_by_id = {record["id"]: record for record in load_golden_set(golden_set_file)}
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in load_jsonl(chunks_file)}
    answer_records = load_jsonl(run_dir / "answers.jsonl")

    # The answer key may be improved after a run (e.g. better expected
    # answers); re-grading is fine as long as the questions are the same.
    changed_questions = [
        record["id"] for record in answer_records
        if record["id"] not in golden_by_id or golden_by_id[record["id"]]["question"] != record["question"]
    ]
    if changed_questions:
        sys.exit(f"Questions changed since this run ({', '.join(changed_questions)}); re-run run_eval.py instead.")

    golden_set_sha256 = file_sha256(golden_set_file)
    if golden_set_sha256 != run_config["golden_set"]["sha256"]:
        print("Note: the answer key changed since this run (questions unchanged); grading with the current version.")

    if args.questions:
        wanted_ids = {question_id.strip() for question_id in args.questions.split(",") if question_id.strip()}
        answer_records = [record for record in answer_records if record["id"] in wanted_ids]
        if len(answer_records) != len(wanted_ids):
            sys.exit("Some --questions IDs are not in this run.")

    # 2. Code checks.
    score_records = []
    scoring_items = []

    for answer_record in answer_records:
        golden_record = golden_by_id[answer_record["id"]]
        score_record = {
            "id": golden_record["id"],
            "type": golden_record["type"],
            "answerable": golden_record["answerable"],
            "error": answer_record.get("error"),
        }

        if score_record["error"] is None:
            score_record["code"] = code_checks(golden_record, answer_record)
            retrieved_texts = [chunks_by_id[item["chunk_id"]]["text"] for item in answer_record["retrieved"]]
            scoring_items.append((golden_record, answer_record, retrieved_texts))

        score_records.append(score_record)

    # 3. Judge checks.
    usage_counter = JudgeUsageCounter()
    judge_started = time.perf_counter()

    if not args.no_judge and scoring_items:
        print(f"Judging {len(scoring_items)} answers with {config.EVAL_JUDGE_MODEL} ...")
        judge_results = asyncio.run(
            judge_checks_for_questions(scoring_items, args.judge_concurrency, usage_counter)
        )
        for score_record in score_records:
            if score_record["id"] in judge_results:
                score_record["judge"] = judge_results[score_record["id"]]

    # 4. Summaries: overall and per question type.
    summary = {
        "run_name": args.run_name,
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions_scored": [record["id"] for record in score_records],
        "golden_set": {
            "file": run_config["golden_set"]["file"],
            "sha256": golden_set_sha256,
            "last_commit": golden_set_last_commit(golden_set_file),
            "same_as_run": golden_set_sha256 == run_config["golden_set"]["sha256"],
        },
        "overall": summarize_score_records(score_records),
        "by_type": {
            question_type: summarize_score_records([r for r in score_records if r["type"] == question_type])
            for question_type in QUESTION_TYPES
            if any(r["type"] == question_type for r in score_records)
        },
        "judge": {
            "model": None if args.no_judge else config.EVAL_JUDGE_MODEL,
            "effort": config.EVAL_JUDGE_EFFORT if config.EVAL_JUDGE_MODEL.startswith("claude-") else None,
            "ragas_version": None if args.no_judge else ragas.__version__,
            "calls": usage_counter.calls,
            "calls_per_model": dict(usage_counter.calls_per_model),
            "token_usage": usage_counter.token_usage,
            "cost_usd": round(usage_counter.cost_usd, 4) if usage_counter.calls and usage_counter.cost_usd is not None else None,
            "seconds": round(time.perf_counter() - judge_started, 1),
        },
        "run_totals": run_config.get("totals", {}),
        "run_settings": run_config.get("settings", {}),
    }

    failures = find_failures(score_records)

    # 5. Save.
    file_prefix = "scores_subset" if args.questions else "scores"
    summary_prefix = "summary_subset" if args.questions else "summary"

    with (run_dir / f"{file_prefix}.jsonl").open("w", encoding="utf-8") as file:
        for score_record in score_records:
            file.write(json.dumps(score_record, ensure_ascii=False) + "\n")

    (run_dir / f"{summary_prefix}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / f"{summary_prefix}.md").write_text(summary_markdown(args.run_name, summary, failures), encoding="utf-8")

    print("\n" + summary_markdown(args.run_name, summary, failures))
    print(f"Saved {file_prefix}.jsonl, {summary_prefix}.json and {summary_prefix}.md in {run_dir}")


if __name__ == "__main__":
    main()
