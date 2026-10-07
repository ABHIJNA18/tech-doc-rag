# ---------------------------------------------------------
# llm_costs.py
#
# Token usage and cost of an LLM call.
#
#   llm_token_usage(message): read the token counts from the raw
#       LangChain message returned by the chat model (input, output,
#       cached input and reasoning tokens)
#   estimate_llm_cost_usd(model, token_usage): turn those counts into
#       US dollars with the prices in config.LLM_PRICES_USD_PER_1M_TOKENS
#
# Used to report the cost per answer in the evaluation (and in the
# README results table).
# ---------------------------------------------------------

from __future__ import annotations

from typing import Any

from rag import config


def llm_token_usage(message: Any) -> dict[str, int]:
    """
    Token counts of one chat model response. OpenAI reports cached
    input tokens (a cheaper rate) and reasoning tokens (already part of
    the output tokens) separately; LangChain puts them in
    usage_metadata. Missing values count as 0.
    """

    usage = getattr(message, "usage_metadata", None) or {}
    input_details = usage.get("input_token_details") or {}
    output_details = usage.get("output_token_details") or {}

    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cached_input_tokens": input_details.get("cache_read", 0) or 0,
        "reasoning_tokens": output_details.get("reasoning", 0) or 0,
    }


def estimate_llm_cost_usd(model: str, token_usage: dict[str, int]) -> float | None:
    """
    Estimated cost of one call in US dollars, or None when the model
    has no price in config.
    """

    prices = config.LLM_PRICES_USD_PER_1M_TOKENS.get(model)

    if prices is None:
        return None

    cached_input_tokens = token_usage.get("cached_input_tokens", 0)
    uncached_input_tokens = token_usage.get("input_tokens", 0) - cached_input_tokens

    cost_per_1m_tokens = (
        uncached_input_tokens * prices["input"]
        + cached_input_tokens * prices["cached_input"]
        + token_usage.get("output_tokens", 0) * prices["output"]
    )

    return cost_per_1m_tokens / 1_000_000
