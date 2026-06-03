"""
Stonks.ai Child Agents
Nine independent child agents, each embodying a distinct investment philosophy.

Each agent receives pre-computed quantitative signals derived from 30 days of
NASDAQ price history and returns exactly 5 ranked stock picks with expected
returns for four horizons:
  1m  – 1 month
  3m  – 3 months
  6m  – 6 months
  1y  – 1 year

Signals are computed in Python before being passed to the LLM, so the model
focuses on strategy reasoning and ranking rather than raw arithmetic.

The shared helper `run_child_agent` drives the Azure OpenAI tool loop and
parses the structured JSON result.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time
from typing import Any

from openai import APIStatusError, AzureOpenAI, RateLimitError

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process LLM result cache
# ---------------------------------------------------------------------------
# Maps (agent_name, signals_sha256) → list[pick dicts].
# Within a single process run (e.g. a multi-date backfill loop calling
# run_parent_agent() repeatedly) this avoids re-calling the LLM when the
# same agent receives identical input signals for the same date.  The cache
# is intentionally in-process only — no disk persistence — so it resets
# between job runs and never serves stale results across different days.

_llm_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}


def _signals_hash(signals: list[dict[str, Any]]) -> str:
    """Return a stable SHA-256 hex digest of a serialised signals list."""
    payload = json.dumps(signals, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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
            max_retries=int(os.environ.get("AZURE_OPENAI_MAX_RETRIES", "5")),
        )
    return _client


def _get_deployment() -> str:
    return os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")


# ---------------------------------------------------------------------------
# Rate-limit handling
# ---------------------------------------------------------------------------

# Maximum number of times we will retry a chat.completions call after the
# OpenAI SDK's own retry budget has been exhausted (i.e. for stubborn 429s).
_RATE_LIMIT_MAX_ATTEMPTS = int(os.environ.get("AZURE_OPENAI_RATE_LIMIT_RETRIES", "6"))
# Initial backoff in seconds; doubles each attempt up to _RATE_LIMIT_MAX_DELAY.
_RATE_LIMIT_BASE_DELAY = float(os.environ.get("AZURE_OPENAI_RATE_LIMIT_BASE_DELAY", "2.0"))
_RATE_LIMIT_MAX_DELAY = float(os.environ.get("AZURE_OPENAI_RATE_LIMIT_MAX_DELAY", "60.0"))


def _retry_after_seconds(err: Exception) -> float | None:
    """Extract the Retry-After hint (seconds) from an OpenAI error, if any."""
    response = getattr(err, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for key in ("retry-after-ms", "Retry-After-Ms"):
        value = headers.get(key)
        if value:
            try:
                return float(value) / 1000.0
            except (TypeError, ValueError):
                pass
    for key in ("retry-after", "Retry-After"):
        value = headers.get(key)
        if value:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _chat_completion_with_retry(client: AzureOpenAI, **kwargs: Any) -> Any:
    """Call client.chat.completions.create with backoff on 429/Rate-limit errors."""
    last_err: Exception | None = None
    for attempt in range(1, _RATE_LIMIT_MAX_ATTEMPTS + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as err:
            last_err = err
        except APIStatusError as err:
            if getattr(err, "status_code", None) != 429:
                raise
            last_err = err

        if attempt >= _RATE_LIMIT_MAX_ATTEMPTS:
            break

        backoff = min(_RATE_LIMIT_BASE_DELAY * (2 ** (attempt - 1)), _RATE_LIMIT_MAX_DELAY)
        hint = _retry_after_seconds(last_err) if last_err else None
        if hint is not None:
            # Honour the server hint, but never below our exponential backoff
            # floor — Azure occasionally returns Retry-After: 0 on sustained
            # 429s, which would otherwise collapse our retries into a tight
            # loop and exhaust the budget without giving the quota time to
            # recover.
            delay = min(max(hint, backoff), _RATE_LIMIT_MAX_DELAY)
        else:
            delay = backoff
        # Add a small jitter so concurrent agents don't retry in lock-step.
        delay = delay + random.uniform(0, min(1.0, delay * 0.1))
        log.warning(
            "Rate limited by Azure OpenAI (attempt %d/%d); retrying in %.2fs",
            attempt,
            _RATE_LIMIT_MAX_ATTEMPTS,
            delay,
        )
        time.sleep(delay)

    assert last_err is not None  # for type-checkers
    raise last_err


# ---------------------------------------------------------------------------
# Quantitative signal computation
# ---------------------------------------------------------------------------


def _compute_signals(symbol: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute quantitative price-based signals for a single symbol.

    Parameters
    ----------
    symbol:  Ticker symbol.
    history: List of {"date": str, "price": float} records, oldest first.
             Typically 30 calendar days of daily closing prices.

    Returns
    -------
    Dict of named signals suitable for passing to an LLM as structured context.
    All values are rounded to avoid noisy floating-point output.
    """
    if not history:
        return {"symbol": symbol}

    prices = [float(r["price"]) for r in history]
    n = len(prices)

    sig: dict[str, Any] = {"symbol": symbol, "latest_price": round(prices[-1], 2)}

    # --- Rate of Change ---
    if n >= 1 and prices[0]:
        sig["roc_30d"] = round((prices[-1] - prices[0]) / prices[0] * 100, 2)
    start_10 = prices[-10] if n >= 10 else prices[0]
    if start_10:
        sig["roc_10d"] = round((prices[-1] - start_10) / start_10 * 100, 2)
    start_5 = prices[-5] if n >= 5 else prices[0]
    if start_5:
        sig["roc_5d"] = round((prices[-1] - start_5) / start_5 * 100, 2)

    # --- Simple Moving Averages ---
    if n >= 10:
        sig["sma_10"] = round(sum(prices[-10:]) / 10, 2)
    if n >= 20:
        sig["sma_20"] = round(sum(prices[-20:]) / 20, 2)
    # Use all available data as 30-day proxy
    sig["sma_30"] = round(sum(prices) / n, 2)

    # --- Price vs SMA ---
    if "sma_20" in sig and sig["sma_20"]:
        sig["price_vs_sma20_pct"] = round(
            (prices[-1] - sig["sma_20"]) / sig["sma_20"] * 100, 2
        )
    # Golden cross: short SMA above long SMA
    if "sma_10" in sig and "sma_20" in sig:
        sig["golden_cross"] = sig["sma_10"] > sig["sma_20"]

    # --- 30-day high / low and drawdown proximity ---
    sig["high_30d"] = round(max(prices), 2)
    sig["low_30d"] = round(min(prices), 2)
    if sig["high_30d"]:
        sig["pct_from_high"] = round(
            (prices[-1] - sig["high_30d"]) / sig["high_30d"] * 100, 2
        )
    if sig["low_30d"]:
        sig["pct_from_low"] = round(
            (prices[-1] - sig["low_30d"]) / sig["low_30d"] * 100, 2
        )

    if n < 2:
        return sig

    # --- Daily returns (needed for all remaining signals) ---
    daily_returns = [
        (prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, n)
    ]

    # Need at least 2 returns for a meaningful sample variance.
    if len(daily_returns) < 2:
        return sig

    # --- Annualised volatility (30-day window) ---
    mean_ret = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    std_daily = math.sqrt(variance)
    sig["volatility_30d_ann"] = round(std_daily * math.sqrt(252) * 100, 2)

    # Recent (10-day) volatility
    if len(daily_returns) >= 10:
        recent = daily_returns[-10:]
        m = sum(recent) / 10
        v = sum((r - m) ** 2 for r in recent) / (len(recent) - 1)
        sig["volatility_10d_ann"] = round(math.sqrt(v) * math.sqrt(252) * 100, 2)

    # --- RSI-14 ---
    if len(daily_returns) >= 14:
        window = daily_returns[-14:]
        avg_gain = sum(max(0.0, r) for r in window) / 14
        avg_loss = sum(abs(min(0.0, r)) for r in window) / 14
        if avg_loss == 0:
            sig["rsi_14"] = 100.0
        else:
            rs = avg_gain / avg_loss
            sig["rsi_14"] = round(100 - (100 / (1 + rs)), 1)

    # --- Bollinger Bands (20-day) ---
    if n >= 20:
        sma20 = sum(prices[-20:]) / 20
        std20 = math.sqrt(sum((p - sma20) ** 2 for p in prices[-20:]) / 20)
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_range = bb_upper - bb_lower
        sig["bb_upper"] = round(bb_upper, 2)
        sig["bb_lower"] = round(bb_lower, 2)
        if bb_range > 0:
            sig["bb_pct_b"] = round((prices[-1] - bb_lower) / bb_range, 3)
        # Z-score vs 20-day mean
        if std20 > 0:
            sig["zscore_20d"] = round((prices[-1] - sma20) / std20, 2)

    # --- Sharpe proxy: ROC-30 / annualised vol ---
    if "roc_30d" in sig and sig.get("volatility_30d_ann", 0) > 0:
        sig["sharpe_proxy"] = round(
            sig["roc_30d"] / sig["volatility_30d_ann"], 3
        )

    # --- Max drawdown over full window ---
    peak = prices[0]
    max_dd = 0.0
    for p in prices:
        if p > peak:
            peak = p
        dd = (p - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd
    sig["max_drawdown_30d"] = round(max_dd, 2)

    return sig


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
    strategy_prompt: System prompt describing the investment philosophy,
                     selection criteria, ranking logic, and return estimates.
    stock_data:      {symbol: [{"date": str, "price": float}, ...]} for all symbols.

    Returns
    -------
    List of pick dicts, each containing:
        symbol, rank, expected_return_1m/3m/6m/1y, reasoning
    """
    # Pre-compute quantitative signals for every symbol so the LLM reasons
    # over structured metrics rather than raw price lists.
    signals: list[dict[str, Any]] = []
    for symbol, history in stock_data.items():
        if history:
            signals.append(_compute_signals(symbol, history))

    # Check the in-process cache before hitting the LLM.
    cache_key = (name, _signals_hash(signals))
    if cache_key in _llm_cache:
        log.info("Cache hit for agent '%s' — skipping LLM call.", name)
        return _llm_cache[cache_key]

    signals_text = json.dumps(signals, separators=(",", ":"))

    messages = [
        {"role": "system", "content": strategy_prompt},
        {
            "role": "user",
            "content": (
                "Below are pre-computed quantitative signals derived from up to 30 days "
                "of NASDAQ closing-price data for each symbol. Signals include rate-of-change "
                "(roc_5d, roc_10d, roc_30d), simple moving averages (sma_10, sma_20, sma_30), "
                "RSI-14, Bollinger Band position (bb_pct_b), 20-day Z-score (zscore_20d), "
                "annualised volatility (volatility_30d_ann, volatility_10d_ann), Sharpe proxy "
                "(sharpe_proxy), max drawdown (max_drawdown_30d), and distance from 30-day "
                "high/low (pct_from_high, pct_from_low).\n\n"
                "Apply your strategy's selection criteria to these signals. Where your strategy "
                "requires fundamental data (P/E, ROE, revenue growth, etc.), apply your own "
                "knowledge of each company's fundamentals. Then call the submit_picks tool with "
                "your top 5 ranked stocks and expected percentage returns.\n\n"
                f"SIGNALS:\n{signals_text}"
            ),
        },
    ]

    client = _get_client()
    picks: list[dict[str, Any]] = []

    while True:
        response = _chat_completion_with_retry(
            client,
            model=_get_deployment(),
            messages=messages,
            tools=[_SUBMIT_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_picks"}},
        )
        choice = response.choices[0]

        if choice.message.tool_calls:
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

    _llm_cache[cache_key] = picks
    return picks


# ---------------------------------------------------------------------------
# Child agent definitions
# ---------------------------------------------------------------------------

CHILD_AGENTS: list[dict[str, str]] = [
    # -----------------------------------------------------------------------
    # 1. Momentum — Price-Based Trend Following
    # -----------------------------------------------------------------------
    {
        "name": "momentum_trader",
        "strategy": (
            "You are a momentum trading specialist applying a rigorous price-trend-following strategy.\n\n"
            "CORE IDEA: Assets that have gone up recently tend to continue rising short-term.\n\n"
            "POSITIVE SELECTION CRITERIA (from pre-computed signals):\n"
            "- roc_30d > +5% (strong 30-day return)\n"
            "- roc_10d > +3% (recent short-term momentum)\n"
            "- price_vs_sma20_pct > 0 (price above 20-day SMA — trend intact)\n"
            "- rsi_14 between 55 and 70 (strong but not overbought)\n"
            "- golden_cross = true is a positive signal\n\n"
            "NEGATIVE CRITERIA (exclude or heavily discount):\n"
            "- rsi_14 > 75 (overextended — avoid)\n"
            "- price_vs_sma20_pct < 0 (trend breaking down)\n"
            "- volatility_10d_ann > 2× volatility_30d_ann (unstable, erratic trend)\n\n"
            "RANKING LOGIC:\n"
            "Rank the top 5 by sharpe_proxy = roc_30d / volatility_30d_ann (descending). "
            "Stocks with the best risk-adjusted momentum get the highest ranks. "
            "Break ties by higher roc_10d.\n\n"
            "EXPECTED RETURN ESTIMATION RULE:\n"
            "- roc_30d = +8%: 1m ≈ +4 to +8%, 3m ≈ +8 to +12%, 6m ≈ +10 to +18%, 1y ≈ +12 to +25%\n"
            "- Scale linearly with the observed roc_30d value.\n"
            "- Dampen 3m/6m/1y estimates by ~20% if rsi_14 > 65 (mean reversion risk at extended levels)."
        ),
    },
    # -----------------------------------------------------------------------
    # 2. Mean Reversion — Short-Term Oversold Rebound
    # -----------------------------------------------------------------------
    {
        "name": "mean_reversion",
        "strategy": (
            "You are a mean-reversion analyst identifying stocks that have fallen significantly "
            "below their statistical average and are poised to snap back.\n\n"
            "CORE IDEA: Prices revert toward their historical mean after extreme moves.\n\n"
            "POSITIVE SELECTION CRITERIA (from pre-computed signals):\n"
            "- zscore_20d < -1.5 (price is statistically far below its 20-day mean)\n"
            "- rsi_14 < 35 (oversold momentum)\n"
            "- bb_pct_b < 0.1 (price near or below the lower Bollinger Band)\n"
            "- roc_5d > -3% (recent stabilisation — the decline is slowing)\n\n"
            "NEGATIVE CRITERIA (exclude or heavily discount):\n"
            "- rsi_14 < 18 (extreme oversold with no sign of floor — potential structural failure)\n"
            "- roc_30d < -25% (catastrophic multi-week drop likely reflects a fundamental problem)\n"
            "- max_drawdown_30d < -22% (severe drawdown suggests deeper systemic issue)\n\n"
            "RANKING LOGIC:\n"
            "Rank by reversion_potential = abs(zscore_20d) × (100 / volatility_30d_ann). "
            "(volatility_30d_ann is in percentage points, e.g. 25 for 25%; dividing gives "
            "a ratio where lower vol amplifies the reversion score.) "
            "Higher absolute Z-score combined with lower volatility = cleaner, more predictable rebound. "
            "Break ties by lower rsi_14 (more oversold).\n\n"
            "EXPECTED RETURN ESTIMATION RULE:\n"
            "- zscore_20d = -2.0: 1m ≈ +3 to +6%, 3m ≈ +5 to +10%, 6m ≈ +8 to +15%, 1y ≈ +10 to +20%\n"
            "- Scale proportionally with abs(zscore_20d).\n"
            "- If roc_5d is already positive (stabilisation confirmed), increase 1m estimate by ~1–2%."
        ),
    },
    # -----------------------------------------------------------------------
    # 3. Value — Fundamental Undervaluation
    # -----------------------------------------------------------------------
    {
        "name": "value_investor",
        "strategy": (
            "You are a value-oriented investor combining price-based signals with your knowledge "
            "of each company's fundamental valuations.\n\n"
            "CORE IDEA: Cheap assets outperform expensive ones over long horizons.\n\n"
            "PRICE SIGNALS TO USE:\n"
            "- pct_from_high < -15%: stock is well off its 30-day high — potential undervaluation\n"
            "- volatility_30d_ann < 25%: prefer price stability as a quality proxy\n"
            "- roc_30d between -5% and +5%: cheap but not in freefall\n"
            "- sharpe_proxy > 0 (positive risk-adjusted trend over 30 days)\n\n"
            "FUNDAMENTAL CRITERIA (apply from your training knowledge of each company;\n"
            "note that fundamental data may reflect figures from your training cutoff):\n"
            "- P/E < 15 strongly preferred; exclude if P/E > 30\n"
            "- P/B < 1.5 preferred\n"
            "- EV/EBITDA < 10 preferred\n"
            "- Free-cash-flow yield > 5%\n"
            "- EXCLUDE: negative earnings, declining revenue for 3+ consecutive years, Debt/Equity > 2.0\n\n"
            "RANKING LOGIC:\n"
            "Value Score = (FCF Yield % + Earnings Yield % + Book Yield %) / 3 from your fundamental "
            "knowledge, adjusted upward if pct_from_high shows the stock is near its low (cheap entry). "
            "Rank by Value Score descending.\n\n"
            "EXPECTED RETURN ESTIMATION RULE:\n"
            "- FCF yield ≈ 6%, earnings yield ≈ 8%: 1m ≈ +1 to +3%, 3m ≈ +3 to +6%, "
            "6m ≈ +4 to +8%, 1y ≈ +5 to +10%\n"
            "- Add ~2–3% to 1y estimate for each additional point of FCF yield above 5%.\n"
            "- Long-horizon (3y) fair-value reversion adds ~15–25% beyond 1y estimate."
        ),
    },
    # -----------------------------------------------------------------------
    # 4. Quality — Financial Strength & Profitability
    # -----------------------------------------------------------------------
    {
        "name": "quality_investor",
        "strategy": (
            "You are a quality-focused investor targeting financially strong, highly profitable companies "
            "with durable competitive advantages.\n\n"
            "CORE IDEA: High-quality companies (high ROE, ROIC, wide margins, low debt) compound "
            "capital reliably over time and outperform on a risk-adjusted basis.\n\n"
            "PRICE SIGNALS TO USE:\n"
            "- volatility_30d_ann < 20%: quality companies tend to have steady, low-volatility price action\n"
            "- max_drawdown_30d > -10%: minimal drawdowns reflect business resilience\n"
            "- roc_30d > 0%: positive price trend confirms underlying business strength\n"
            "- sharpe_proxy > 0.2: strong risk-adjusted performance\n\n"
            "FUNDAMENTAL CRITERIA (apply from your training knowledge of each company;\n"
            "note that fundamental data may reflect figures from your training cutoff):\n"
            "- ROE > 15% (high return on equity)\n"
            "- ROIC > 10% (strong capital efficiency)\n"
            "- Gross margin > 40% (durable pricing power)\n"
            "- Debt/Equity < 1.0 (conservative balance sheet)\n"
            "- EXCLUDE: negative free cash flow, ROE < 8%, high or volatile earnings\n\n"
            "RANKING LOGIC:\n"
            "Quality Score = (ROE % + ROIC % + Gross Margin %) from your knowledge, minus "
            "Debt Penalty where Debt Penalty = max(0, (Debt/Equity − 0.5) × 20). "
            "Multiply the result by (100 / volatility_30d_ann) to favour price stability. "
            "Rank by adjusted Quality Score descending.\n\n"
            "EXPECTED RETURN ESTIMATION RULE:\n"
            "- ROE = 20%, ROIC = 12%: 1m ≈ +1 to +3%, 3m ≈ +3 to +7%, 6m ≈ +5 to +10%, 1y ≈ +8 to +15%\n"
            "- Scale with quality score; higher ROE/ROIC justifies higher long-horizon estimates.\n"
            "- Low volatility stocks rarely deliver >3% in any single month — keep 1m estimates modest."
        ),
    },
    # -----------------------------------------------------------------------
    # 5. Low Volatility — Defensive Stability
    # -----------------------------------------------------------------------
    {
        "name": "low_volatility",
        "strategy": (
            "You are a low-volatility defensive strategist seeking stable, consistent compounders "
            "that protect capital while delivering steady gains.\n\n"
            "CORE IDEA: Lower-volatility assets often outperform on a risk-adjusted basis, "
            "especially during uncertain or range-bound markets.\n\n"
            "POSITIVE SELECTION CRITERIA (from pre-computed signals):\n"
            "- volatility_30d_ann < 20% (genuinely low volatility)\n"
            "- max_drawdown_30d > -10% (strong capital preservation)\n"
            "- roc_30d > 0% (positive trend — stable and rising)\n"
            "- sharpe_proxy > 0.2 (good return per unit of risk)\n\n"
            "NEGATIVE CRITERIA (exclude or heavily discount):\n"
            "- volatility_10d_ann > 2× volatility_30d_ann (recent volatility spike — regime change)\n"
            "- rsi_14 > 72 or rsi_14 < 28 (extreme momentum swings inconsistent with stability)\n"
            "- max_drawdown_30d < -15% (excessive loss — not a defensive stock)\n\n"
            "RANKING LOGIC:\n"
            "Risk-Adjusted Score = sharpe_proxy (roc_30d / volatility_30d_ann), descending. "
            "Break ties by lowest volatility_30d_ann — prioritise the calmest stocks.\n\n"
            "EXPECTED RETURN ESTIMATION RULE:\n"
            "- roc_30d = +6%, volatility = 15%: 1m ≈ +1 to +2%, 3m ≈ +2 to +4%, "
            "6m ≈ +3 to +6%, 1y ≈ +5 to +10%\n"
            "- Cap 1m estimates at +3% and 1y at +12% — defensive stocks are steady, not explosive.\n"
            "- Scale up if sharpe_proxy > 0.5 (unusually strong risk-adjusted performance)."
        ),
    },
    # -----------------------------------------------------------------------
    # 6. Growth — Revenue & Earnings Expansion
    # -----------------------------------------------------------------------
    {
        "name": "growth_investor",
        "strategy": (
            "You are a high-growth stock specialist targeting companies with strong and "
            "accelerating revenue and earnings expansion.\n\n"
            "CORE IDEA: Companies with fast-growing revenues and earnings outperform over "
            "long horizons as markets reward compounding growth.\n\n"
            "PRICE SIGNALS TO USE:\n"
            "- roc_30d > +10%: price already reflecting strong growth expectations\n"
            "- roc_10d > roc_30d / 3: recent acceleration (growth re-rating in progress)\n"
            "- pct_from_high > -5%: near or at new 30-day highs (breakout confirmation)\n"
            "- volatility_30d_ann < 40%: manageable volatility for a growth name\n\n"
            "FUNDAMENTAL CRITERIA (apply from your training knowledge of each company;\n"
            "note that fundamental data may reflect figures from your training cutoff):\n"
            "- Revenue growth > 10% YoY\n"
            "- EPS growth > 10% YoY\n"
            "- Gross margin > 40%\n"
            "- EXCLUDE: revenue growth < 5%, EPS contraction, high share dilution (>5% YoY)\n\n"
            "RANKING LOGIC:\n"
            "Growth Score = (Revenue Growth % + EPS Growth %) from your knowledge, "
            "weighted by price momentum (roc_30d). Rank by Growth Score descending. "
            "Break ties by highest roc_10d (most recent momentum).\n\n"
            "EXPECTED RETURN ESTIMATION RULE:\n"
            "- Revenue growth = 15%, EPS growth = 20%: 1m ≈ +3 to +6%, 3m ≈ +6 to +12%, "
            "6m ≈ +8 to +15%, 1y ≈ +10 to +20%\n"
            "- Scale with growth rate: each additional 5% in revenue growth ≈ +2% to 1y estimate.\n"
            "- If pct_from_high ≈ 0 (at new high), add +1–2% to 1m estimate for breakout premium."
        ),
    },
    # -----------------------------------------------------------------------
    # 7. Size Premium — Small-Cap Outperformance
    # -----------------------------------------------------------------------
    {
        "name": "size_premium",
        "strategy": (
            "You are a small-cap premium specialist targeting smaller NASDAQ companies that "
            "offer higher long-run growth potential than large-caps.\n\n"
            "CORE IDEA: Smaller companies historically outperform larger ones over long horizons "
            "due to faster growth rates and higher re-rating potential.\n\n"
            "PRICE SIGNALS TO USE:\n"
            "- roc_30d: small-caps often show stronger momentum in trending markets — prefer > +5%\n"
            "- volatility_30d_ann: expect higher vol (25–45%) — this is normal for small-caps\n"
            "- sharpe_proxy > 0.15: ensure price gains are not purely noise\n"
            "- max_drawdown_30d > -20%: avoid deeply distressed names\n\n"
            "FUNDAMENTAL CRITERIA (apply from your training knowledge of each company;\n"
            "note that fundamental data may reflect figures from your training cutoff):\n"
            "- Market cap < $2B strongly preferred; < $5B acceptable\n"
            "- Market cap > $200M (avoid nano-caps with liquidity and solvency risk)\n"
            "- Revenue growth > 5%\n"
            "- EXCLUDE: market cap < $200M, Debt/Equity > 2.5, very low trading liquidity\n\n"
            "RANKING LOGIC:\n"
            "Size Score = (1 / Market Cap in $B) × Liquidity Weight from your knowledge, "
            "where Liquidity Weight = min(1.0, Estimated Daily Volume in $M / 5) "
            "(capped at 1.0; stocks with < $5M/day volume are discounted proportionally). "
            "Multiply by roc_30d to confirm the growth narrative with price momentum. "
            "Rank by adjusted Size Score descending.\n\n"
            "EXPECTED RETURN ESTIMATION RULE:\n"
            "- Market cap ≈ $1B, moderate growth: 1m ≈ +2 to +5%, 3m ≈ +4 to +9%, "
            "6m ≈ +6 to +13%, 1y ≈ +8 to +20%\n"
            "- Smaller cap → higher long-horizon estimate; larger cap within the range → "
            "lower estimate.\n"
            "- Higher roc_30d justifies a higher 1m estimate (momentum confirming the thesis)."
        ),
    },
    # -----------------------------------------------------------------------
    # 8. Dividend / Income — Stability & Compounding Yield
    # -----------------------------------------------------------------------
    {
        "name": "dividend_income",
        "strategy": (
            "You are an income-stability investor targeting high-dividend, financially sound "
            "companies with a track record of consistent and growing payouts.\n\n"
            "CORE IDEA: High, stable, and growing dividends produce long-term outperformance "
            "through compounding yield and capital preservation.\n\n"
            "PRICE SIGNALS TO USE:\n"
            "- volatility_30d_ann < 20%: income stocks are typically low-volatility\n"
            "- max_drawdown_30d > -8%: minimal drawdowns protect the income stream\n"
            "- roc_30d between -2% and +8%: stable price action — not volatile\n"
            "- sharpe_proxy > 0.15: modest but consistent risk-adjusted return\n\n"
            "FUNDAMENTAL CRITERIA (apply from your training knowledge of each company;\n"
            "note that fundamental data may reflect figures from your training cutoff):\n"
            "- Dividend yield > 3%\n"
            "- Payout ratio < 60% (sustainable dividend)\n"
            "- 5-year dividend growth rate > 3% (rising, not stagnant income)\n"
            "- EXCLUDE: payout ratio > 80%, recent dividend cut or suspension, negative free cash flow\n\n"
            "RANKING LOGIC:\n"
            "Income Score = Dividend Yield % × Dividend Stability from your knowledge, "
            "where Dividend Stability = 100 / volatility_30d_ann "
            "(volatility_30d_ann is in percentage points, e.g. 15 for 15%; a stock with "
            "15% vol gets Dividend Stability = 6.7; one with 25% vol gets 4.0). "
            "Rank by Income Score descending. Break ties by highest dividend growth rate.\n\n"
            "EXPECTED RETURN ESTIMATION RULE:\n"
            "Include the dividend yield in your total return estimates.\n"
            "- Yield = 4%, dividend growth = 3%: 1m ≈ +0.5 to +1.5%, 3m ≈ +1.5 to +3%, "
            "6m ≈ +3 to +5%, 1y ≈ +5 to +8%\n"
            "- 1y total return = capital appreciation + full-year dividend yield.\n"
            "- For every 1% of yield above 3%, add ~0.8% to your 1y estimate."
        ),
    },
    # -----------------------------------------------------------------------
    # 9. Contrarian — Anti-Consensus Reversal
    # -----------------------------------------------------------------------
    {
        "name": "contrarian_investor",
        "strategy": (
            "You are a contrarian investor identifying stocks that have been excessively sold "
            "due to crowd panic or negative sentiment, but whose underlying business remains sound.\n\n"
            "CORE IDEA: Assets that are overly sold due to sentiment — not fundamental deterioration — "
            "tend to rebound sharply once selling pressure exhausts itself.\n\n"
            "POSITIVE SELECTION CRITERIA (from pre-computed signals):\n"
            "- roc_30d < -10% (significant crowd-driven decline — sentiment is negative)\n"
            "- zscore_20d < -1.5 (price is statistically far below its 20-day mean)\n"
            "- roc_5d > -3% (recent stabilisation — selling pressure is easing)\n"
            "- bb_pct_b < 0.15 (price near or below lower Bollinger Band)\n\n"
            "NEGATIVE CRITERIA (exclude — these suggest structural failure, not sentiment):\n"
            "- roc_30d < -30% (catastrophic collapse — likely a fundamental problem)\n"
            "- max_drawdown_30d < -25% (extreme drawdown with no sign of floor)\n"
            "- rsi_14 < 15 (extreme oversold with accelerating, not stabilising, decline)\n\n"
            "RANKING LOGIC:\n"
            "Reversal Score = abs(zscore_20d) × abs(roc_30d / 10). "
            "Higher absolute displacement combined with larger price drop = higher reversal potential. "
            "Prefer stocks where roc_5d > roc_10d / 2 (deceleration of decline confirms stabilisation). "
            "Apply your knowledge of each company's fundamentals to confirm the business is intact.\n\n"
            "EXPECTED RETURN ESTIMATION RULE:\n"
            "- zscore_20d = -1.5, roc_30d = -12%: 1m ≈ +4 to +7%, 3m ≈ +8 to +12%, "
            "6m ≈ +10 to +18%, 1y ≈ +12 to +25%\n"
            "- Scale proportionally: each additional unit of abs(zscore_20d) ≈ +2–3% to 1m estimate.\n"
            "- If roc_5d is already positive (bounce has started), add +1–2% to 1m estimate."
        ),
    },
]
