# ---------------------------------------------------------
# tracing.py
#
# Langfuse tracing in one place. Langfuse records every request as a
# trace (a timeline of steps with inputs, outputs and timing) and
# sends it in the background to the Langfuse dashboard.
#
# Tracing is optional: when the Langfuse keys are not set (tests,
# offline evaluation, a deployment without Langfuse), `observe` is a
# decorator that does nothing and the code runs exactly the same.
#
# Keys are read from the environment / .env:
#   LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
# ---------------------------------------------------------

from __future__ import annotations

import os
from typing import Any, Callable

from rag import config  # noqa: F401  (loads .env before the keys are read)

TRACING_ENABLED = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))

if TRACING_ENABLED:
    from langfuse import get_client, observe
else:

    def observe(func: Callable | None = None, **_: Any):  # type: ignore[no-redef]
        """
        No-op stand-in for langfuse.observe, usable as @observe and
        @observe(name=...).
        """

        if func is not None:
            return func

        return lambda f: f


def update_current_span(**fields: Any) -> None:
    """
    Add output or metadata to the step currently being traced.
    """

    if TRACING_ENABLED:
        get_client().update_current_span(**fields)


def current_trace_id() -> str | None:
    return get_client().get_current_trace_id() if TRACING_ENABLED else None


def flush() -> None:
    """
    Send pending traces now. Needed at the end of short scripts,
    because traces are sent in the background.
    """

    if TRACING_ENABLED:
        get_client().flush()
