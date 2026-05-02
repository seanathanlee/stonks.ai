"""
Stonks.ai FastAPI web application.

Serves the chat frontend and exposes the REST API used by the browser UI.

Usage:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter setup
#
# Uses an in-memory sliding-window store. This is sufficient for single-
# instance deployments. For multi-instance / production environments replace
# storage_uri with a Redis URL, e.g.:
#   storage_uri = os.environ.get("REDIS_URL", "memory://")
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Stonks.ai", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Allow cross-origin requests from the configured origins.
# CORS_ORIGINS is a comma-separated list of allowed origins; defaults to the
# wildcard only when the env var is not set (e.g. local development).
_raw_cors = os.environ.get("CORS_ORIGINS", "")
_cors_origins: list[str] = [o.strip() for o in _raw_cors.split(",") if o.strip()] or ["*"]
if _cors_origins == ["*"]:
    log.warning(
        "CORS_ORIGINS is not set — allowing all origins. "
        "Set CORS_ORIGINS to a comma-separated list of allowed origins in production."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve the frontend directory as static files
_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(_FRONTEND_DIR / "index.html")


# ---------------------------------------------------------------------------
# In-memory session store
# Keyed by session_id → list of conversation messages (role/content dicts)
# ---------------------------------------------------------------------------

_sessions: dict[str, list[dict[str, Any]]] = {}

_MAX_SESSIONS = 500  # evict oldest when limit reached


def _get_or_create_session(session_id: str) -> list[dict[str, Any]]:
    if session_id not in _sessions:
        if len(_sessions) >= _MAX_SESSIONS:
            # Evict the oldest entry
            oldest = next(iter(_sessions))
            del _sessions[oldest]
        _sessions[session_id] = []
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128, description="Client-generated session UUID.")
    message: str = Field(..., min_length=1, max_length=2000, description="User message text.")

    @field_validator("session_id")
    @classmethod
    def _validate_session_id(cls, v: str) -> str:
        # Only allow alphanumeric characters, hyphens, and underscores
        if not re.fullmatch(r"[A-Za-z0-9_-]+", v):
            raise ValueError("session_id contains invalid characters")
        return v


class ChartDataset(BaseModel):
    label: str
    data: list[float | None]
    backgroundColor: str | list[str] | None = None
    borderColor: str | None = None
    borderRadius: int | None = None
    tension: float | None = None
    fill: bool | None = None
    spanGaps: bool | None = None


class ChartData(BaseModel):
    id: str
    type: str
    title: str
    labels: list[str]
    datasets: list[dict[str, Any]]
    indexAxis: str | None = None


class ChatResponse(BaseModel):
    reply: str
    charts: list[dict[str, Any]] = []
    session_id: str
    redirect_url: str | None = None


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    """
    Process a user message and return the assistant reply plus optional charts.
    """
    from agents.chat_agent import run_chat

    history = _get_or_create_session(req.session_id)

    try:
        reply, updated_history, charts, redirect_url = run_chat(req.message, history)
    except Exception as exc:
        log.exception("Chat agent error for session %s", req.session_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _sessions[req.session_id] = updated_history

    return ChatResponse(reply=reply, charts=charts, session_id=req.session_id, redirect_url=redirect_url)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/api/health")
@limiter.limit("60/minute")
async def health(request: Request) -> dict[str, str]:
    return {"status": "ok"}
