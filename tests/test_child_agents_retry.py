"""Regression tests for the Azure OpenAI rate-limit retry helper."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest
from openai import RateLimitError

from agents import child_agents


def _make_rate_limit_error(retry_after: str | None = None) -> RateLimitError:
    """Build a RateLimitError with an optional Retry-After header."""
    headers = {}
    if retry_after is not None:
        headers["retry-after"] = retry_after
    request = httpx.Request("POST", "https://example.invalid/chat")
    response = httpx.Response(429, headers=headers, request=request)
    return RateLimitError("rate limited", response=response, body=None)


class _StubClient:
    """Minimal stand-in for AzureOpenAI exposing chat.completions.create."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls = 0

        outer = self

        class _Completions:
            def create(self, **kwargs: Any) -> Any:  # noqa: ANN401
                outer.calls += 1
                value = outer._responses.pop(0)
                if isinstance(value, Exception):
                    raise value
                return value

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_zero_retry_after_does_not_skip_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `Retry-After: 0` header must not collapse our backoff to zero seconds.

    Previously the wrapper trusted the server hint blindly, so Azure returning
    `retry-after: 0` on sustained 429s caused us to sleep 0s between retries
    and burn through the budget instantly. The hint is now floored at the
    exponential-backoff value.
    """
    monkeypatch.setattr(child_agents, "_RATE_LIMIT_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(child_agents, "_RATE_LIMIT_BASE_DELAY", 4.0)
    monkeypatch.setattr(child_agents, "_RATE_LIMIT_MAX_DELAY", 60.0)
    # Drop jitter to keep the assertion exact.
    monkeypatch.setattr(child_agents.random, "uniform", lambda _a, _b: 0.0)

    client = _StubClient(
        [
            _make_rate_limit_error("0"),
            _make_rate_limit_error("0"),
            "ok",
        ]
    )

    sleeps: list[float] = []
    with patch.object(child_agents.time, "sleep", side_effect=sleeps.append):
        result = child_agents._chat_completion_with_retry(client)  # type: ignore[arg-type]

    assert result == "ok"
    assert client.calls == 3
    # First retry floor = 4.0 * 2**0 = 4.0; second = 4.0 * 2**1 = 8.0.
    assert sleeps == [4.0, 8.0]


def test_server_hint_above_floor_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    """When Azure suggests a longer delay than our backoff, we honour it."""
    monkeypatch.setattr(child_agents, "_RATE_LIMIT_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(child_agents, "_RATE_LIMIT_BASE_DELAY", 1.0)
    monkeypatch.setattr(child_agents, "_RATE_LIMIT_MAX_DELAY", 60.0)
    monkeypatch.setattr(child_agents.random, "uniform", lambda _a, _b: 0.0)

    client = _StubClient(
        [
            _make_rate_limit_error("12"),
            "ok",
        ]
    )

    sleeps: list[float] = []
    with patch.object(child_agents.time, "sleep", side_effect=sleeps.append):
        result = child_agents._chat_completion_with_retry(client)  # type: ignore[arg-type]

    assert result == "ok"
    assert sleeps == [12.0]
