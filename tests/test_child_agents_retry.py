"""Regression tests for the Azure OpenAI rate-limit retry helper and LLM cache."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# LLM result cache tests
# ---------------------------------------------------------------------------


def _make_stock_data() -> dict[str, list[dict[str, Any]]]:
    return {
        "AAPL": [{"date": "2025-01-01", "price": 150.0 + i} for i in range(30)],
        "MSFT": [{"date": "2025-01-01", "price": 300.0 + i} for i in range(30)],
    }


def _stub_llm_response(picks: list[dict[str, Any]]) -> MagicMock:
    """Build a minimal mock response that looks like a tool-call completion."""
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.arguments = __import__("json").dumps({"picks": picks})

    message = MagicMock()
    message.tool_calls = [tool_call]

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


def test_cache_hit_skips_llm_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second call with identical signals must be served from cache."""
    monkeypatch.setattr(child_agents, "_llm_cache", {})

    picks = [
        {
            "rank": i,
            "symbol": "AAPL",
            "expected_return_1m": 1.0,
            "expected_return_3m": 2.0,
            "expected_return_6m": 3.0,
            "expected_return_1y": 4.0,
            "reasoning": "test",
        }
        for i in range(1, 6)
    ]
    llm_response = _stub_llm_response(picks)

    stock_data = _make_stock_data()

    call_count = 0

    def fake_retry(client: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return llm_response

    with (
        patch.object(child_agents, "_chat_completion_with_retry", fake_retry),
        patch.object(child_agents, "_get_client", return_value=MagicMock()),
        patch.object(child_agents, "_get_deployment", return_value="gpt-4.1"),
    ):
        result1 = child_agents.run_child_agent("test_agent", "strategy", stock_data)
        result2 = child_agents.run_child_agent("test_agent", "strategy", stock_data)

    assert call_count == 1, "LLM should only be called once; second call should hit cache"
    assert result1 == result2


def test_different_stock_data_bypasses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Different input signals must result in a fresh LLM call."""
    monkeypatch.setattr(child_agents, "_llm_cache", {})

    picks = [
        {
            "rank": i,
            "symbol": "AAPL",
            "expected_return_1m": 1.0,
            "expected_return_3m": 2.0,
            "expected_return_6m": 3.0,
            "expected_return_1y": 4.0,
            "reasoning": "test",
        }
        for i in range(1, 6)
    ]

    call_count = 0

    def fake_retry(client: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return _stub_llm_response(picks)

    stock_data_a = _make_stock_data()
    stock_data_b = {
        "AAPL": [{"date": "2025-01-01", "price": 200.0 + i} for i in range(30)],
        "MSFT": [{"date": "2025-01-01", "price": 400.0 + i} for i in range(30)],
    }

    with (
        patch.object(child_agents, "_chat_completion_with_retry", fake_retry),
        patch.object(child_agents, "_get_client", return_value=MagicMock()),
        patch.object(child_agents, "_get_deployment", return_value="gpt-4.1"),
    ):
        child_agents.run_child_agent("test_agent", "strategy", stock_data_a)
        child_agents.run_child_agent("test_agent", "strategy", stock_data_b)

    assert call_count == 2, "Different signals should trigger two separate LLM calls"


def test_different_agents_bypass_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same signals but different agent names must each get their own LLM call."""
    monkeypatch.setattr(child_agents, "_llm_cache", {})

    picks = [
        {
            "rank": i,
            "symbol": "AAPL",
            "expected_return_1m": 1.0,
            "expected_return_3m": 2.0,
            "expected_return_6m": 3.0,
            "expected_return_1y": 4.0,
            "reasoning": "test",
        }
        for i in range(1, 6)
    ]

    call_count = 0

    def fake_retry(client: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return _stub_llm_response(picks)

    stock_data = _make_stock_data()

    with (
        patch.object(child_agents, "_chat_completion_with_retry", fake_retry),
        patch.object(child_agents, "_get_client", return_value=MagicMock()),
        patch.object(child_agents, "_get_deployment", return_value="gpt-4.1"),
    ):
        child_agents.run_child_agent("agent_a", "strategy", stock_data)
        child_agents.run_child_agent("agent_b", "strategy", stock_data)

    assert call_count == 2, "Different agent names must each call the LLM independently"
