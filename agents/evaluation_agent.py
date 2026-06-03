"""
Stonks.ai Evaluation Agent
Evaluates how accurate each agent's stock forecasts were, comparing
predictions made N days ago against realized returns.

Supports all four forecast horizons:
  1m  – forecast made 30 days ago, evaluated against 30-day realized return
  3m  – forecast made 91 days ago, evaluated against 91-day realized return
  6m  – forecast made 182 days ago, evaluated against 182-day realized return
  1y  – forecast made 365 days ago, evaluated against 365-day realized return

Workflow:
  1. Determine the forecast date (today − LOOKBACK_DAYS for the chosen horizon).
  2. Query all agent forecasts stored on that date for the chosen horizon.
  3. Fetch actual closing prices for those symbols over the window.
  4. Compute realized returns, rank symbols by actual performance, and
     calculate per-agent accuracy metrics:
       - accuracyScore        (legacy composite error, lower = better)
       - returnMAE            (|forecastReturn − actualReturn|)
       - returnBias           (forecastReturn − actualReturn, signed)
       - directionCorrect     (1 if direction matched, 0 otherwise)
       - volatilityAdjustedError (returnMAE / annualised realized volatility)
       - spearmanRho          (rank correlation between forecast & actual ordering)
       - precisionAt5         (fraction of agent's top-5 in actual top-5)
  5. Persist results to the ADX `agentStockEvaluation` table.

Usage:
  python -m agents.evaluation_agent
  python -m agents.evaluation_agent --date 2025-01-15
  python -m agents.evaluation_agent --date 2025-01-15 --horizon 3m
"""

from __future__ import annotations

import argparse
import logging
import math
import uuid
from datetime import date, timedelta, timezone, datetime
from typing import Any

from agents.adx_client import (
    get_forecasts_from_date,
    get_price_range,
    ingest_evaluations,
    now_utc_iso,
)
from agents.horizons import ALL_HORIZONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Mapping of forecast horizon label → calendar-day lookback window.
# Imported from agents.horizons so that adding a new horizon only requires
# a change in one place.
HORIZON_LOOKBACK_DAYS: dict[str, int] = ALL_HORIZONS

# Default horizon evaluated when none is specified.
DEFAULT_HORIZON = "1m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _forecast_date(horizon: str, today: date | None = None) -> str:
    """Return the ISO-8601 date string for the forecast lookback start of *horizon*."""
    lookback = HORIZON_LOOKBACK_DAYS[horizon]
    ref = today or date.today()
    return (ref - timedelta(days=lookback)).isoformat()


def _compute_actual_return(start_price: float, end_price: float) -> float:
    """Realized percentage return: (end − start) / start × 100."""
    if start_price == 0:
        return 0.0
    return round((end_price - start_price) / start_price * 100, 4)


def _compute_realized_volatility(prices: list[dict[str, Any]]) -> float | None:
    """
    Annualised realized volatility as a percentage (std dev of daily log
    returns × sqrt(252) × 100).

    Returns None when fewer than 3 data points are available.
    """
    if len(prices) < 3:
        return None
    ps = [r["price"] for r in prices]
    log_rets = [
        math.log(ps[i] / ps[i - 1])
        for i in range(1, len(ps))
        if ps[i - 1] > 0 and ps[i] > 0
    ]
    if len(log_rets) < 2:
        return None
    n = len(log_rets)
    mean = sum(log_rets) / n
    variance = sum((r - mean) ** 2 for r in log_rets) / (n - 1)
    return round(math.sqrt(variance) * math.sqrt(252) * 100.0, 4)


def _spearman_rho(
    forecast_ranks: list[int],
    actual_ranks: list[int],
) -> float | None:
    """
    Spearman rank correlation between two equal-length rank lists.

    Uses the general Pearson-on-ranks formula so tied ranks are handled
    correctly.  Returns None when fewer than 2 data points are supplied or
    when either rank list has zero variance (all identical ranks).
    """
    n = len(forecast_ranks)
    if n < 2 or n != len(actual_ranks):
        return None
    fr_mean = sum(forecast_ranks) / n
    ar_mean = sum(actual_ranks) / n
    num = sum(
        (f - fr_mean) * (a - ar_mean)
        for f, a in zip(forecast_ranks, actual_ranks)
    )
    denom_fr = math.sqrt(sum((f - fr_mean) ** 2 for f in forecast_ranks))
    denom_ar = math.sqrt(sum((a - ar_mean) ** 2 for a in actual_ranks))
    if denom_fr < 1e-10 or denom_ar < 1e-10:
        return None
    return round(num / (denom_fr * denom_ar), 4)


def _assign_actual_ranks(
    symbol_returns: dict[str, float],
) -> dict[str, int]:
    """
    Rank symbols by descending actual return (rank 1 = best performer).
    Ties receive the same rank (dense ranking).
    """
    sorted_symbols = sorted(
        symbol_returns.keys(),
        key=lambda s: symbol_returns[s],
        reverse=True,
    )
    ranks: dict[str, int] = {}
    current_rank = 1
    prev_value: float | None = None
    for i, symbol in enumerate(sorted_symbols):
        value = symbol_returns[symbol]
        if prev_value is not None and value != prev_value:
            current_rank = i + 1
        ranks[symbol] = current_rank
        prev_value = value
    return ranks


def _build_evaluation_rows(
    forecasts: list[dict[str, Any]],
    actual_returns: dict[str, float],
    actual_ranks: dict[str, int],
    symbol_volatilities: dict[str, float | None],
    report_time: str,
    run_id: str,
) -> list[dict[str, Any]]:
    """
    Build evaluation rows ready for ADX ingestion.

    Each row includes:
        accuracyScore          – legacy composite error (lower = better)
        returnMAE              – |forecastReturn − actualReturn|
        returnBias             – forecastReturn − actualReturn (signed)
        directionCorrect       – 1 if direction matched, 0 otherwise
        volatilityAdjustedError – returnMAE / annualised volatility (or null)

    The per-run metrics spearmanRho and precisionAt5 are added in a
    subsequent pass by _enrich_rows_with_run_metrics.
    """
    rows: list[dict[str, Any]] = []
    for f in forecasts:
        symbol = f["symbol"]
        if symbol not in actual_returns:
            log.warning("No actual price data for symbol %s — skipping.", symbol)
            continue

        forecast_return = f["expectedReturn"]
        actual_return = actual_returns[symbol]
        forecast_rank = f["rank"]
        actual_rank = actual_ranks.get(symbol, 0)

        # Legacy combined error metric (lower = better).
        abs_return_error = abs(forecast_return - actual_return)
        abs_rank_error = abs(forecast_rank - actual_rank)
        accuracy_score = round((abs_return_error + abs_rank_error) / 2, 4)

        # Directional accuracy: did the agent predict the correct sign?
        # Both-zero is treated as a correct call (no move → no move).
        forecast_positive = forecast_return > 0
        actual_positive = actual_return > 0
        forecast_negative = forecast_return < 0
        actual_negative = actual_return < 0
        direction_correct = int(
            (forecast_positive and actual_positive)
            or (forecast_negative and actual_negative)
            or (forecast_return == 0.0 and actual_return == 0.0)
        )

        # Return MAE and signed bias.
        return_mae = round(abs_return_error, 4)
        return_bias = round(forecast_return - actual_return, 4)

        # Volatility-adjusted error (None if volatility is unavailable/zero).
        vol = symbol_volatilities.get(symbol)
        if vol and vol > 1e-6:
            vol_adj_error: float | None = round(return_mae / vol, 4)
        else:
            vol_adj_error = None

        rows.append(
            {
                "symbol": symbol,
                "forecastReturn": forecast_return,
                "actualReturn": actual_return,
                "forecastRank": forecast_rank,
                "actualRank": actual_rank,
                "accuracyScore": accuracy_score,
                "returnMAE": return_mae,
                "returnBias": return_bias,
                "directionCorrect": direction_correct,
                "volatilityAdjustedError": vol_adj_error,
                # spearmanRho and precisionAt5 are filled by
                # _enrich_rows_with_run_metrics after all rows are built.
                "spearmanRho": None,
                "precisionAt5": None,
                "agentName": f["agentName"],
                "forecastReportTime": f["reportTime"],
                "reportTime": report_time,
                "runId": run_id,
                "horizon": f["horizon"],
            }
        )
    return rows


def _enrich_rows_with_run_metrics(
    rows: list[dict[str, Any]],
    actual_ranks: dict[str, int],
) -> None:
    """
    Compute per-agent-per-run Spearman ρ and Precision@5 and write them
    back onto every row in-place.

    spearmanRho:
        Rank correlation between the agent's predicted ordering and the
        actual return ordering, computed over the agent's own picked symbols.

    precisionAt5:
        Fraction of the agent's top-5 picks that appear in the actual top-5
        performers from the full universe (i.e. symbols with actual_rank ≤ 5).
    """
    # Actual top-5 symbols from the full universe.
    actual_top5 = {sym for sym, rank in actual_ranks.items() if rank <= 5}

    # Group rows by agent name (within this run there is a single runId).
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_agent.setdefault(row["agentName"], []).append(row)

    for agent_name, agent_rows in by_agent.items():
        # --- Spearman ρ ---
        forecast_ranks = [r["forecastRank"] for r in agent_rows]
        agent_actual_ranks = [r["actualRank"] for r in agent_rows]
        rho = _spearman_rho(forecast_ranks, agent_actual_ranks)

        # --- Precision@5 ---
        agent_symbols = {r["symbol"] for r in agent_rows}
        overlap = len(agent_symbols & actual_top5)
        precision = round(overlap / 5, 4)

        for row in agent_rows:
            row["spearmanRho"] = rho
            row["precisionAt5"] = precision


def _print_summary(
    rows: list[dict[str, Any]],
    forecast_date: str,
    report_time: str,
    horizon: str,
) -> None:
    """Print a human-readable evaluation summary."""
    print("\n" + "=" * 80)
    print(f"Stonks.ai Evaluation Report  |  {report_time}")
    print(f"Horizon: {horizon}  |  Forecasts evaluated from: {forecast_date}")
    print(f"Rows evaluated: {len(rows)}")
    print("=" * 80)

    if not rows:
        print("  No rows to display.")
        print("=" * 80 + "\n")
        return

    # Aggregate per-agent metrics.
    agent_data: dict[str, dict[str, list]] = {}
    for row in rows:
        a = row["agentName"]
        if a not in agent_data:
            agent_data[a] = {
                "accuracyScore": [],
                "returnMAE": [],
                "returnBias": [],
                "directionCorrect": [],
                "spearmanRho": [],
                "precisionAt5": [],
            }
        agent_data[a]["accuracyScore"].append(row["accuracyScore"])
        agent_data[a]["returnMAE"].append(row["returnMAE"])
        agent_data[a]["returnBias"].append(row["returnBias"])
        agent_data[a]["directionCorrect"].append(row["directionCorrect"])
        if row.get("spearmanRho") is not None:
            agent_data[a]["spearmanRho"].append(row["spearmanRho"])
        if row.get("precisionAt5") is not None:
            agent_data[a]["precisionAt5"].append(row["precisionAt5"])

    header = (
        f"\n  {'Agent':<28} {'AccScore':>9} {'MAE':>8} {'Bias':>8}"
        f" {'HitRate':>8} {'SpearRho':>9} {'P@5':>6} {'#':>4}"
    )
    divider = "  " + "-" * (len(header) - 2)
    print(header)
    print(divider)

    def _mean(lst: list) -> float:
        return sum(lst) / len(lst) if lst else float("nan")

    sorted_agents = sorted(
        agent_data.items(),
        key=lambda kv: _mean(kv[1]["accuracyScore"]),
    )
    for agent, d in sorted_agents:
        rho_vals = d["spearmanRho"]
        p5_vals = d["precisionAt5"]
        rho_str = f"{_mean(rho_vals):>9.4f}" if rho_vals else f"{'N/A':>9}"
        p5_str = f"{_mean(p5_vals):>6.4f}" if p5_vals else f"{'N/A':>6}"
        print(
            f"  {agent:<28}"
            f" {_mean(d['accuracyScore']):>9.4f}"
            f" {_mean(d['returnMAE']):>8.4f}"
            f" {_mean(d['returnBias']):>8.4f}"
            f" {_mean(d['directionCorrect']):>8.4f}"
            f" {rho_str}"
            f" {p5_str}"
            f" {len(d['accuracyScore']):>4}"
        )

    print("\n" + "=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_evaluation_agent(
    today: date | None = None,
    horizon: str = DEFAULT_HORIZON,
) -> list[dict[str, Any]]:
    """
    Execute the full evaluation pipeline and return all evaluation rows.

    Args:
        today:   Override today's date (useful for testing). Defaults to UTC today.
        horizon: Forecast horizon to evaluate. Must be one of "1m", "3m", "6m",
                 "1y". Defaults to "1m".
    """
    if horizon not in HORIZON_LOOKBACK_DAYS:
        raise ValueError(
            f"Invalid horizon {horizon!r}. Must be one of "
            + ", ".join(HORIZON_LOOKBACK_DAYS)
        )

    report_time = now_utc_iso()
    run_id = str(uuid.uuid4())
    forecast_date = _forecast_date(horizon, today)

    log.info(
        "Evaluation run %s started. Horizon=%s, evaluating forecasts from %s.",
        run_id,
        horizon,
        forecast_date,
    )

    # ------------------------------------------------------------------
    # 1. Query forecasts from the appropriate lookback date
    # ------------------------------------------------------------------
    log.info("Querying %s-horizon forecasts from %s…", horizon, forecast_date)
    forecasts = get_forecasts_from_date(forecast_date, horizon=horizon)

    if not forecasts:
        log.warning(
            "No forecasts found for date %s with horizon '%s'. "
            "Ensure the parent agent ran on that date.",
            forecast_date,
            horizon,
        )
        return []

    symbols = sorted({f["symbol"] for f in forecasts})
    log.info("Found %d forecast rows covering %d symbols.", len(forecasts), len(symbols))

    # ------------------------------------------------------------------
    # 2. Pull actual prices over the evaluation window
    # ------------------------------------------------------------------
    today_date = today or date.today()
    log.info(
        "Fetching actual prices for %d symbols from %s to %s…",
        len(symbols),
        forecast_date,
        today_date.isoformat(),
    )
    price_data = get_price_range(symbols, start_date=forecast_date, end_date=today_date.isoformat())

    # ------------------------------------------------------------------
    # 3. Compute realized returns and per-symbol volatility
    # ------------------------------------------------------------------
    actual_returns: dict[str, float] = {}
    symbol_volatilities: dict[str, float | None] = {}

    for symbol in symbols:
        prices = price_data.get(symbol, [])
        if len(prices) < 2:
            log.warning(
                "Insufficient price data for %s (%d data points) — skipping.",
                symbol,
                len(prices),
            )
            continue
        start_price = prices[0]["price"]
        end_price = prices[-1]["price"]
        actual_returns[symbol] = _compute_actual_return(start_price, end_price)
        symbol_volatilities[symbol] = _compute_realized_volatility(prices)

    log.info("Computed actual returns for %d symbols.", len(actual_returns))

    # ------------------------------------------------------------------
    # 4. Assign actual ranks and compute per-prediction accuracy metrics
    # ------------------------------------------------------------------
    actual_ranks = _assign_actual_ranks(actual_returns)
    evaluation_rows = _build_evaluation_rows(
        forecasts,
        actual_returns,
        actual_ranks,
        symbol_volatilities,
        report_time,
        run_id,
    )

    # ------------------------------------------------------------------
    # 5. Enrich rows with per-agent-per-run metrics (Spearman ρ, P@5)
    # ------------------------------------------------------------------
    _enrich_rows_with_run_metrics(evaluation_rows, actual_ranks)

    # ------------------------------------------------------------------
    # 6. Persist to ADX
    # ------------------------------------------------------------------
    if evaluation_rows:
        log.info("Ingesting %d evaluation rows into ADX…", len(evaluation_rows))
        ingest_evaluations(evaluation_rows)
        log.info("Evaluation ingest complete.")
    else:
        log.warning("No evaluation rows produced — nothing to ingest.")

    _print_summary(evaluation_rows, forecast_date, report_time, horizon)

    return evaluation_rows


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stonks.ai evaluation agent")
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help=(
            "Treat this ISO-8601 date (YYYY-MM-DD) as 'today' when evaluating "
            "forecasts. Defaults to today's UTC date. Useful for backfilling."
        ),
    )
    parser.add_argument(
        "--horizon",
        choices=list(HORIZON_LOOKBACK_DAYS),
        default=DEFAULT_HORIZON,
        help=(
            "Forecast horizon to evaluate. "
            "1m=30 days, 3m=91 days, 6m=182 days, 1y=365 days. "
            f"Defaults to {DEFAULT_HORIZON!r}."
        ),
    )
    args = parser.parse_args()
    run_evaluation_agent(today=args.date, horizon=args.horizon)
