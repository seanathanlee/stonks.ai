"""
Stonks.ai FastAPI web application.

Serves the chat frontend and exposes the REST API used by the browser UI.

Usage:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Stonks.ai", version="1.0.0")

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
    session_id: str = Field(..., description="Client-generated session UUID.")
    message: str = Field(..., min_length=1, description="User message text.")


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


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Process a user message and return the assistant reply plus optional charts.
    """
    from agents.chat_agent import run_chat

    history = _get_or_create_session(req.session_id)

    try:
        reply, updated_history, charts = run_chat(req.message, history)
    except Exception as exc:
        log.exception("Chat agent error for session %s", req.session_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _sessions[req.session_id] = updated_history

    return ChatResponse(reply=reply, charts=charts, session_id=req.session_id)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
