"""
Stonks.ai Evaluation Agent
Evaluates how accurate each agent's 1-month stock forecasts were,
comparing predictions made 30 days ago against realized returns.

Workflow:
  1. Determine the forecast date (today − 30 days).
  2. Query all agent forecasts stored on that date (horizon = "1m").
  3. Fetch actual closing prices for those symbols over the 30-day window.
  4. Compute realized returns, rank symbols by actual performance, and
     calculate per-agent accuracy scores.
  5. Persist results to the ADX `agentStockEvaluation` table.

Usage:
  python -m agents.evaluation_agent
  python -m agents.evaluation_agent --date 2025-01-15
"""

from __future__ import annotations

import argparse
import logging
import uuid
from datetime import date, timedelta, timezone, datetime
from typing import Any

from agents.adx_client import (
    get_forecasts_from_date,
    get_price_range,
    ingest_evaluations,
    now_utc_iso,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EVALUATION_HORIZON = "1m"
LOOKBACK_DAYS = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _forecast_date(today: date | None = None) -> str:
    """Return the ISO-8601 date string for exactly LOOKBACK_DAYS ago."""
    ref = today or date.today()
    return (ref - timedelta(days=LOOKBACK_DAYS)).isoformat()


def _compute_actual_return(start_price: float, end_price: float) -> float:
    """Realized percentage return: (end − start) / start × 100."""
    if start_price == 0:
        return 0.0
    return round((end_price - start_price) / start_price * 100, 4)


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
    report_time: str,
    run_id: str,
) -> list[dict[str, Any]]:
    """Build evaluation rows ready for ADX ingestion."""
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

        # accuracy_score: combined error metric (lower = better).
        # Computed as the average of absolute return error and absolute rank error.
        abs_return_error = abs(forecast_return - actual_return)
        abs_rank_error = abs(forecast_rank - actual_rank)
        accuracy_score = round((abs_return_error + abs_rank_error) / 2, 4)

        rows.append(
            {
                "symbol": symbol,
                "forecastReturn": forecast_return,
                "actualReturn": actual_return,
                "forecastRank": forecast_rank,
                "actualRank": actual_rank,
                "accuracyScore": accuracy_score,
                "agentName": f["agentName"],
                "forecastReportTime": f["reportTime"],
                "reportTime": report_time,
                "runId": run_id,
                "horizon": f["horizon"],
            }
        )
    return rows


def _print_summary(
    rows: list[dict[str, Any]],
    forecast_date: str,
    report_time: str,
) -> None:
    """Print a human-readable evaluation summary."""
    print("\n" + "=" * 70)
    print(f"Stonks.ai Evaluation Report  |  {report_time}")
    print(f"Forecasts evaluated from: {forecast_date}")
    print(f"Rows evaluated: {len(rows)}")
    print("=" * 70)

    if not rows:
        print("  No rows to display.")
        print("=" * 70 + "\n")
        return

    # Per-agent mean accuracy score (lower = better)
    agent_scores: dict[str, list[float]] = {}
    for row in rows:
        agent_scores.setdefault(row["agentName"], []).append(row["accuracyScore"])

    print(f"\n{'Agent':<30} {'Mean Accuracy Score':>20} {'# Predictions':>15}")
    print(f"{'-'*30:<30} {'-'*20:>20} {'-'*15:>15}")
    for agent, scores in sorted(agent_scores.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        mean = sum(scores) / len(scores)
        print(f"  {agent:<28} {mean:>20.4f} {len(scores):>15}")

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_evaluation_agent(today: date | None = None) -> list[dict[str, Any]]:
    """
    Execute the full evaluation pipeline and return all evaluation rows.

    Args:
        today: Override today's date (useful for testing). Defaults to UTC today.
    """
    report_time = now_utc_iso()
    run_id = str(uuid.uuid4())
    forecast_date = _forecast_date(today)

    log.info(
        "Evaluation run %s started. Evaluating forecasts from %s.",
        run_id,
        forecast_date,
    )

    # ------------------------------------------------------------------
    # 1. Query forecasts from 30 days ago
    # ------------------------------------------------------------------
    log.info("Querying %s-horizon forecasts from %s…", EVALUATION_HORIZON, forecast_date)
    forecasts = get_forecasts_from_date(forecast_date, horizon=EVALUATION_HORIZON)

    if not forecasts:
        log.warning(
            "No forecasts found for date %s with horizon '%s'. "
            "Ensure the parent agent ran on that date.",
            forecast_date,
            EVALUATION_HORIZON,
        )
        return []

    symbols = sorted({f["symbol"] for f in forecasts})
    log.info("Found %d forecast rows covering %d symbols.", len(forecasts), len(symbols))

    # ------------------------------------------------------------------
    # 2. Pull actual prices over the 30-day window
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
    # 3. Compute realized returns for each symbol
    # ------------------------------------------------------------------
    actual_returns: dict[str, float] = {}
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

    log.info("Computed actual returns for %d symbols.", len(actual_returns))

    # ------------------------------------------------------------------
    # 4. Assign actual ranks and compute accuracy metrics
    # ------------------------------------------------------------------
    actual_ranks = _assign_actual_ranks(actual_returns)
    evaluation_rows = _build_evaluation_rows(
        forecasts, actual_returns, actual_ranks, report_time, run_id
    )

    # ------------------------------------------------------------------
    # 5. Persist to ADX
    # ------------------------------------------------------------------
    if evaluation_rows:
        log.info("Ingesting %d evaluation rows into ADX…", len(evaluation_rows))
        ingest_evaluations(evaluation_rows)
        log.info("Evaluation ingest complete.")
    else:
        log.warning("No evaluation rows produced — nothing to ingest.")

    _print_summary(evaluation_rows, forecast_date, report_time)

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
    args = parser.parse_args()
    run_evaluation_agent(today=args.date)
