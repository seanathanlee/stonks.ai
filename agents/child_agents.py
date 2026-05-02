"""
Stonks.ai Child Agents
Nine independent child agents, each embodying a distinct investment strategy.

Each agent receives 30 days of NASDAQ price history and returns exactly 5
ranked stock picks with expected returns for four horizons:
  1m  – 1 month
  3m  – 3 months
  6m  – 6 months
  1y  – 1 year

The shared helper `run_child_agent` drives the Azure OpenAI tool loop and
parses the structured JSON result.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import AzureOpenAI

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------

_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
    return _client


def _get_deployment() -> str:
    return os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


# ---------------------------------------------------------------------------
# Shared agent runner
# ---------------------------------------------------------------------------

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "picks": {
            "type": "array",
            "description": "Exactly 5 stock picks, ranked 1 (best) to 5.",
            "items": {
                "type": "object",
                "properties": {
                    "rank": {"type": "integer"},
                    "symbol": {"type": "string"},
                    "expected_return_1m": {
                        "type": "number",
                        "description": "Expected % return over 1 month.",
                    },
                    "expected_return_3m": {
                        "type": "number",
                        "description": "Expected % return over 3 months.",
                    },
                    "expected_return_6m": {
                        "type": "number",
                        "description": "Expected % return over 6 months.",
                    },
                    "expected_return_1y": {
                        "type": "number",
                        "description": "Expected % return over 1 year.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief rationale for this pick.",
                    },
                },
                "required": [
                    "rank",
                    "symbol",
                    "expected_return_1m",
                    "expected_return_3m",
                    "expected_return_6m",
                    "expected_return_1y",
                    "reasoning",
                ],
            },
            "minItems": 5,
            "maxItems": 5,
        }
    },
    "required": ["picks"],
}

_SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_picks",
        "description": (
            "Submit exactly 5 ranked stock picks with expected returns for "
            "1-month, 3-month, 6-month, and 1-year horizons."
        ),
        "parameters": _RESPONSE_SCHEMA,
    },
}


def run_child_agent(
    name: str,
    strategy_prompt: str,
    stock_data: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Run a single child agent and return its 5 picks.

    Parameters
    ----------
    name:            Human-readable agent name (stored in forecasts).
    strategy_prompt: System prompt describing the investment philosophy.
    stock_data:      {symbol: [{"date": str, "price": float}, ...]} for all symbols.

    Returns
    -------
    List of pick dicts, each containing:
        symbol, rank, expected_return_1m/3m/6m/1y, reasoning
    """
    # Summarise price data as compact JSON to keep the prompt manageable.
    # Each symbol contributes its most recent 10 data points.
    summary: dict[str, Any] = {}
    for symbol, history in stock_data.items():
        recent = history[-10:] if len(history) > 10 else history
        if recent:
            prices = [r["price"] for r in recent]
            summary[symbol] = {
                "latest_price": prices[-1],
                "price_30d_ago": history[0]["price"] if history else None,
                "recent_prices": prices,
            }

    data_text = json.dumps(summary, separators=(",", ":"))

    messages = [
        {"role": "system", "content": strategy_prompt},
        {
            "role": "user",
            "content": (
                "Below is 30-day NASDAQ price data (latest 10 prices per symbol shown). "
                "Analyse it and use the submit_picks tool to return your top 5 stock picks "
                "ranked to maximise profits, with expected percentage returns for the "
                "1-month, 3-month, 6-month, and 1-year horizons.\n\n"
                f"PRICE DATA:\n{data_text}"
            ),
        },
    ]

    client = _get_client()
    picks: list[dict[str, Any]] = []

    while True:
        response = client.chat.completions.create(
            model=_get_deployment(),
            messages=messages,
            tools=[_SUBMIT_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_picks"}},
        )
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            tool_call = choice.message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            picks = args.get("picks", [])
            # Acknowledge the tool call so the loop can end cleanly
            messages.append(choice.message)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({"status": "received"}),
                }
            )
            break

        # If the model stopped without calling the tool, try once more
        if picks:
            break
        messages.append(choice.message)
        messages.append(
            {
                "role": "user",
                "content": "Please call the submit_picks tool with your 5 picks.",
            }
        )

    return picks


# ---------------------------------------------------------------------------
# Child agent definitions
# ---------------------------------------------------------------------------

CHILD_AGENTS: list[dict[str, str]] = [
    {
        "name": "momentum_trader",
        "strategy": (
            "You are a momentum trading specialist. You identify stocks with the "
            "strongest recent upward price momentum and buy into trends that are "
            "already moving. Focus on relative strength, rate of change, and "
            "breakouts above recent highs. Favour stocks showing accelerating gains."
        ),
    },
    {
        "name": "mean_reversion",
        "strategy": (
            "You are a mean-reversion analyst. You identify stocks that have fallen "
            "significantly below their recent averages and are likely to bounce back. "
            "Look for oversold conditions, support levels, and statistical deviation "
            "from the 30-day moving average. Favour high-quality companies temporarily "
            "depressed."
        ),
    },
    {
        "name": "value_investor",
        "strategy": (
            "You are a value-oriented fundamental investor. You look for stocks that "
            "appear undervalued relative to their historical price ranges. Prioritise "
            "stability, low volatility, and a track record of solid performance. Favour "
            "blue-chip and large-cap NASDAQ names trading near multi-month lows."
        ),
    },
    {
        "name": "growth_investor",
        "strategy": (
            "You are a high-growth stock specialist. You target companies that are "
            "growing rapidly and whose stock price reflects strong market expectations. "
            "Look for consistent upward price trends, high relative volume, and stocks "
            "making new highs. Favour technology, biotech, and consumer-growth names."
        ),
    },
    {
        "name": "volatility_hunter",
        "strategy": (
            "You are a volatility-seeking trader. You look for stocks with the highest "
            "short-term price swings and position yourself to capture outsized moves. "
            "Prioritise stocks with large intraday or day-over-day swings, as these "
            "offer the greatest profit potential when timed correctly."
        ),
    },
    {
        "name": "sector_rotation",
        "strategy": (
            "You are a sector-rotation strategist. You identify which NASDAQ sectors "
            "are gaining momentum and pick the leading stocks within those sectors. "
            "Rotate into outperforming sectors and out of lagging ones. Look at "
            "cross-sector price trends to find where money is flowing."
        ),
    },
    {
        "name": "technical_analyst",
        "strategy": (
            "You are a technical analysis expert. You use price patterns, support and "
            "resistance levels, and chart formations to identify high-probability trade "
            "setups. Look for cup-and-handle patterns, ascending triangles, golden "
            "crosses, and other classic setups in the 30-day price data."
        ),
    },
    {
        "name": "contrarian_investor",
        "strategy": (
            "You are a contrarian investor. You take positions opposite to the prevailing "
            "market sentiment. When the crowd is selling, you buy the most beaten-down "
            "names. Identify stocks with the largest recent declines that have strong "
            "underlying businesses and are likely to recover strongly."
        ),
    },
    {
        "name": "risk_adjusted_optimizer",
        "strategy": (
            "You are a risk-adjusted return optimiser. You seek the best Sharpe-ratio "
            "opportunities — maximising expected return per unit of risk. Evaluate the "
            "recent price volatility of each stock alongside its momentum, and favour "
            "stocks offering strong gains with relatively stable price action."
        ),
    },
]
