"""
Shared fixtures for Playwright end-to-end tests.

A lightweight HTTP server serves ``frontend/index.html``.  All calls to
``/api/*`` are intercepted by Playwright's ``page.route`` so the tests
never need a running FastAPI instance.
"""
from __future__ import annotations

import functools
import http.server
import json
import threading
from pathlib import Path

import pytest
from playwright.sync_api import Page

_FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"

# ---------------------------------------------------------------------------
# Session-scoped HTTP server for the static frontend
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def frontend_url() -> str:  # type: ignore[return]
    """Start a one-shot HTTP server and return its base URL."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(_FRONTEND_DIR),
    )
    # Port 0 → OS picks a free port
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}"

    server.shutdown()


# ---------------------------------------------------------------------------
# Per-test helpers
# ---------------------------------------------------------------------------


def _mock_chat_route(route, reply: str = "Mock reply", charts: list | None = None):
    """Fulfill a /api/chat request with a canned JSON response."""
    route.fulfill(
        status=200,
        content_type="application/json",
        body=json.dumps(
            {"reply": reply, "charts": charts or [], "session_id": "test-session"}
        ),
    )


@pytest.fixture
def page_loaded(page: Page, frontend_url: str) -> Page:
    """
    Navigate to the frontend and wait for the welcome message to appear.
    The ``/api/chat`` route is mocked with a default success response.
    """
    page.route(
        "**/api/chat",
        lambda route: _mock_chat_route(route),
    )
    page.goto(frontend_url)
    # Wait until the DOMContentLoaded welcome message is rendered
    page.wait_for_selector(".message.bot .bubble", timeout=5_000)
    return page
