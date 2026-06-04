"""
Stonks.ai Chat Agent
Conversational AI interface that coordinates the multi-agent pipeline,
queries ADX for forecast data, and returns structured chart payloads.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from openai import AzureOpenAI

from agents.horizons import ALL_HORIZONS

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
    return os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5.1")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Stonks.ai, an intelligent financial analysis assistant powered by a \
multi-agent AI system that analyses NASDAQ stocks.

A pipeline of 9 specialised child agents runs automatically every day and after \
each manual data snapshot to produce fresh stock forecasts:
• momentum_trader         – Buys into strong upward momentum and breakouts; \
tracks relative strength and rate of change.
• mean_reversion          – Finds stocks that have fallen below their 30-day \
average and are likely to bounce back.
• value_investor          – Seeks undervalued blue-chip stocks near multi-month \
lows with low volatility.
• growth_investor         – Targets high-growth technology, biotech, and \
consumer names making new highs.
• volatility_hunter       – Seeks large intraday or day-over-day price swings \
for outsized profit potential.
• sector_rotation         – Rotates into outperforming NASDAQ sectors by \
tracking cross-sector price momentum.
• technical_analyst       – Reads cup-and-handle patterns, ascending triangles, \
golden crosses, and other chart setups.
• contrarian_investor     – Buys beaten-down names against the crowd when \
fundamentals remain strong.
• risk_adjusted_optimizer – Maximises Sharpe ratio, balancing expected return \
against recent price volatility.

Agent evaluations measure each agent's prediction error — how far their \
forecasts were from actual stock returns. A lower error means the agent was \
closer to the real outcome (i.e. more accurate). Use get_agent_evaluations \
to surface this data with visualisations.

Your responsibilities:
1. Understand what the user wants — view existing forecasts, explore specific \
stocks, compare agent opinions, check agent accuracy, or get investment suggestions.
2. Ask at most ONE clarifying question at a time when more context is needed.
3. Use the available tools to fetch real forecast, price, and evaluation data, \
then present it clearly with actionable suggestions.
4. Highlight consensus across agents and explain what each strategy means in \
plain language.
5. Proactively suggest stocks to consider based on the latest forecast data.
6. Investment horizons available: 1m (1 month), 3m (3 months), 6m (6 months), 1y (1 year).

Keep responses concise and insightful. Never fabricate data — always use the tools.
"""

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_latest_forecasts",
            "description": (
                "Retrieve the latest stock forecasts from the database, aggregated across "
                "all child agents. Returns top stocks ranked by average expected return."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "horizon": {
                        "type": "string",
                        "enum": list(ALL_HORIZONS),
                        "description": "Investment horizon to filter by (default: 1m).",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top picks to return (default 5, max 10).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_history",
            "description": "Get historical daily closing prices for specific stock symbols.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock ticker symbols (e.g. ['AAPL', 'MSFT']).",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Days of price history to return (default 30, max 90).",
                    },
                },
                "required": ["symbols"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_agent_forecasts",
            "description": (
                "Compare how all 9 child agents forecast a specific stock symbol. "
                "Shows each agent's expected return broken down by investment horizon."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. 'AAPL').",
                    },
                    "horizon": {
                        "type": "string",
                        "enum": list(ALL_HORIZONS),
                        "description": "Optional: filter to a single horizon.",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_agent_evaluations",
            "description": (
                "Retrieve prediction error metrics for each of the 9 child agents, "
                "showing how closely their forecasts matched actual stock returns. "
                "The prediction error is lower-is-better: a lower score means the "
                "agent's forecasts were closer to the actual returns and rank. "
                "Use this to identify which agents have been most accurate recently."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "horizon": {
                        "type": "string",
                        "enum": list(ALL_HORIZONS),
                        "description": "Forecast horizon to evaluate (default: 1m).",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Days of evaluation history to include (default: 30, max: 90).",
                    },
                },
                "required": [],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

_CHART_COLORS = [
    "#3b82f6",  # blue-500
    "#22c55e",  # green-500
    "#f59e0b",  # amber-500
    "#ec4899",  # pink-500
    "#8b5cf6",  # violet-500
    "#14b8a6",  # teal-500
    "#f97316",  # orange-500
    "#64748b",  # slate-500
    "#a855f7",  # purple-500
]


def _bar_color(value: float) -> str:
    return "#22c55e" if value >= 0 else "#ef4444"


def _build_forecast_chart(forecasts: list[dict[str, Any]], horizon: str) -> dict[str, Any]:
    """Horizontal bar chart: top picks for a horizon."""
    symbols = [f["symbol"] for f in forecasts]
    returns = [f["avgReturn"] for f in forecasts]
    return {
        "id": f"forecast_{horizon}_{uuid.uuid4().hex[:8]}",
        "type": "bar",
        "title": f"Top Picks — {horizon} Horizon (avg across agents)",
        "indexAxis": "y",
        "labels": symbols,
        "datasets": [
            {
                "label": "Avg Expected Return (%)",
                "data": returns,
                "backgroundColor": [_bar_color(r) for r in returns],
                "borderRadius": 4,
            }
        ],
    }


def _build_price_chart(
    price_data: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Multi-line chart: price history per symbol."""
    # Collect union of all dates, sorted
    all_dates: set[str] = set()
    for records in price_data.values():
        for rec in records:
            all_dates.add(rec["date"])
    labels = sorted(all_dates)

    datasets = []
    for idx, (symbol, records) in enumerate(price_data.items()):
        date_to_price = {rec["date"]: rec["price"] for rec in records}
        data = [date_to_price.get(d) for d in labels]
        color = _CHART_COLORS[idx % len(_CHART_COLORS)]
        datasets.append(
            {
                "label": symbol,
                "data": data,
                "borderColor": color,
                "backgroundColor": color + "33",
                "tension": 0.3,
                "fill": False,
                "spanGaps": True,
            }
        )

    return {
        "id": f"price_{uuid.uuid4().hex[:8]}",
        "type": "line",
        "title": "Price History (30 days)",
        "labels": labels,
        "datasets": datasets,
    }


def _build_agent_comparison_chart(
    comparison: list[dict[str, Any]], symbol: str, horizon: str | None
) -> dict[str, Any]:
    """Grouped bar chart: per-agent expected return for a symbol."""
    horizons = sorted({r["horizon"] for r in comparison})
    agents = sorted({r["agentName"] for r in comparison})

    # Build a lookup for quick access
    lookup: dict[tuple[str, str], float] = {
        (r["agentName"], r["horizon"]): r["expectedReturn"] for r in comparison
    }

    datasets = []
    for idx, h in enumerate(horizons):
        data = [lookup.get((a, h)) for a in agents]
        color = _CHART_COLORS[idx % len(_CHART_COLORS)]
        datasets.append(
            {
                "label": h,
                "data": data,
                "backgroundColor": color + "cc",
                "borderRadius": 4,
            }
        )

    title = f"Agent Forecasts — {symbol}"
    if horizon:
        title += f" ({horizon})"

    return {
        "id": f"agent_compare_{symbol}_{uuid.uuid4().hex[:8]}",
        "type": "bar",
        "title": title,
        "labels": agents,
        "datasets": datasets,
    }


def _build_evaluation_chart(
    evaluations: list[dict[str, Any]], horizon: str
) -> dict[str, Any]:
    """Horizontal bar chart: per-agent prediction error (lower = more accurate)."""
    agents = [e["agentName"].replace("_", " ").title() for e in evaluations]
    errors = [e["avgAccuracyScore"] for e in evaluations]
    return {
        "id": f"eval_{horizon}_{uuid.uuid4().hex[:8]}",
        "type": "bar",
        "title": f"Agent Prediction Error — {horizon} Horizon (lower = better)",
        "indexAxis": "y",
        "labels": agents,
        "datasets": [
            {
                "label": "Avg Prediction Error",
                "data": errors,
                "backgroundColor": [
                    _CHART_COLORS[i % len(_CHART_COLORS)] for i in range(len(errors))
                ],
                "borderRadius": 4,
            }
        ],
    }





def _dispatch_tool(name: str, arguments: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Execute a tool call and return (result_text, charts).
    """
    args: dict[str, Any] = json.loads(arguments) if arguments else {}
    charts: list[dict[str, Any]] = []

    if name == "get_latest_forecasts":
        from agents.adx_client import get_latest_forecasts

        horizon = args.get("horizon", "1m")
        top_n = min(int(args.get("top_n", 5)), 10)
        try:
            forecasts = get_latest_forecasts(horizon=horizon, top_n=top_n)
            if forecasts:
                charts.append(_build_forecast_chart(forecasts, horizon))
            result = {"forecasts": forecasts, "horizon": horizon}
        except Exception as exc:
            log.error("get_latest_forecasts failed: %s", exc)
            result = {"error": str(exc)}

    elif name == "get_price_history":
        from agents.adx_client import get_price_history

        symbols: list[str] = [s.upper() for s in args.get("symbols", [])]
        days = min(int(args.get("days", 30)), 90)
        if not symbols:
            result = {"error": "No symbols provided."}
        else:
            try:
                price_data = get_price_history(symbols, days=days)
                non_empty = {s: v for s, v in price_data.items() if v}
                if non_empty:
                    charts.append(_build_price_chart(non_empty))
                result = {
                    "symbols_found": list(non_empty.keys()),
                    "symbols_missing": [s for s in symbols if not price_data.get(s)],
                    "days": days,
                    "record_counts": {s: len(v) for s, v in non_empty.items()},
                }
            except Exception as exc:
                log.error("get_price_history failed: %s", exc)
                result = {"error": str(exc)}

    elif name == "compare_agent_forecasts":
        from agents.adx_client import get_agent_comparison

        symbol = args.get("symbol", "").upper()
        horizon = args.get("horizon")
        if not symbol:
            result = {"error": "No symbol provided."}
        else:
            try:
                comparison = get_agent_comparison(symbol, horizon=horizon)
                if comparison:
                    charts.append(
                        _build_agent_comparison_chart(comparison, symbol, horizon)
                    )
                result = {"symbol": symbol, "comparison": comparison}
            except Exception as exc:
                log.error("compare_agent_forecasts failed: %s", exc)
                result = {"error": str(exc)}

    elif name == "get_agent_evaluations":
        from agents.adx_client import get_agent_evaluations

        horizon = args.get("horizon", "1m")
        days = min(int(args.get("days", 30)), 90)
        try:
            evaluations = get_agent_evaluations(horizon=horizon, days=days)
            if evaluations:
                charts.append(_build_evaluation_chart(evaluations, horizon))
            result = {"evaluations": evaluations, "horizon": horizon, "days": days}
        except Exception as exc:
            log.error("get_agent_evaluations failed: %s", exc)
            result = {"error": str(exc)}

    else:
        result = {"error": f"Unknown tool: {name}"}

    return json.dumps(result), charts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MAX_HISTORY = 20  # maximum messages to retain per session


_BENJI_QUESTION = "Are you Benji?"
_BENJI_REDIRECT_URL = "https://www.youtube.com/shorts/_6HzLIJPH2A"
_BENJI_YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "affirmative", "correct", "indeed", "totally", "absolutely"}


def _is_yes(text: str) -> bool:
    """Return True if the user's response looks like a 'yes'."""
    normalized = text.strip().lower().rstrip("!.?")
    return normalized in _BENJI_YES_WORDS or normalized.startswith("yes") or normalized.startswith("yeah") or normalized.startswith("i am")


def run_chat(
    message: str,
    history: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """
    Process a single user message within an ongoing conversation.

    Parameters
    ----------
    message:  The latest user message.
    history:  Prior conversation messages (role/content dicts).

    Returns
    -------
    (reply, updated_history, charts, redirect_url)
        reply          – the assistant's text response
        updated_history – the conversation history with the new exchange appended
        charts         – list of Chart.js-compatible chart dicts to render
        redirect_url   – optional URL to redirect the client to
    """
    # ── Benji check ──────────────────────────────────────────────────────────
    # After the very first user message ask if it's Benji.
    if len(history) == 0:
        reply = _BENJI_QUESTION
        updated_history = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ]
        return reply, updated_history, [], None

    # If the previous assistant turn was the Benji question, handle the answer.
    if (
        len(history) >= 2
        and history[-1].get("role") == "assistant"
        and history[-1].get("content") == _BENJI_QUESTION
    ):
        if _is_yes(message):
            reply = f"Haha, caught you Benji! 🎉 Redirecting you now…"
            updated_history = list(history)
            updated_history.append({"role": "user", "content": message})
            updated_history.append({"role": "assistant", "content": reply})
            return reply, updated_history, [], _BENJI_REDIRECT_URL

        # Not Benji — replay the original first message through the LLM
        original_message = history[0].get("content", message)
        # Discard the Benji exchange; start fresh so the LLM sees a clean session
        history = []
        message = original_message

    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-MAX_HISTORY:])
    messages.append({"role": "user", "content": message})

    client = _get_client()
    all_charts: list[dict[str, Any]] = []

    while True:
        response = client.chat.completions.create(
            model=_get_deployment(),
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            # Append assistant message with tool calls
            messages.append(choice.message)
            # Execute each tool
            for tool_call in choice.message.tool_calls:
                tool_result, charts = _dispatch_tool(
                    tool_call.function.name,
                    tool_call.function.arguments,
                )
                all_charts.extend(charts)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )
            # Continue loop so model can incorporate results
            continue

        # finish_reason == "stop"
        reply = choice.message.content or ""
        break

    # Build updated history (exclude system prompt)
    updated_history = list(history)
    updated_history.append({"role": "user", "content": message})
    updated_history.append({"role": "assistant", "content": reply})

    return reply, updated_history, all_charts, None
