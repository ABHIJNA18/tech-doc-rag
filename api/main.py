# ---------------------------------------------------------
# api/main.py
#
# The web API for the React frontend (FastAPI).
#
#   GET  /api/health   is the server up, which settings are active
#   POST /api/ask      {"question": "..."} -> answer with citations
#
# /api/ask calls the same answer_question() as the command line, so
# the UI, the CLI and the evaluation all use one RAG pipeline. The
# response shape matches AskResponse in frontend/src/types.ts.
#
# Run locally (from the repository root):
#   uvicorn api.main:app --reload --port 8000
# API docs while running: http://localhost:8000/docs
# ---------------------------------------------------------

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make src/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import config  # noqa: E402
from rag.pipeline import answer_question  # noqa: E402
from rag.retrieval import get_retriever  # noqa: E402
from rag.tracing import TRACING_ENABLED, flush  # noqa: E402

MAX_QUESTION_LENGTH = 1000


@asynccontextmanager
async def api_lifespan(app: FastAPI):
    """
    On startup: open Chroma and set up the retriever once, so the first
    question is not slower. On shutdown: send remaining traces.
    """

    get_retriever(config.RETRIEVAL_MODE)
    yield
    flush()


app = FastAPI(title="Tech Doc RAG API", lifespan=api_lifespan)

# Allow the React app (local dev server, later the Vercel domain) to
# call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)


@app.get("/api/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "retrievalMode": config.RETRIEVAL_MODE,
        "topK": config.TOP_K,
        "llmModel": config.LLM_MODEL,
        "promptVersion": config.PROMPT_VERSION,
        "tracing": TRACING_ENABLED,
    }


@app.post("/api/ask")
def ask_endpoint(request: AskRequest) -> dict:
    """
    Answer a question from the frontend. A plain (non-async) function:
    FastAPI runs it in a worker thread, so slow LLM calls do not block
    other requests.
    """

    result = answer_question(request.question.strip(), source="api")

    return result.to_api_response()
