"""
Stonks.ai Parent Agent (Orchestrator)
Coordinates all 9 child agents and persists results to Azure Data Explorer.

Workflow:
  1. Pull the last 30 days of NASDAQ price data from the ADX `dailyStockPrice` table.
  2. Fan out to all 9 child agents concurrently (ThreadPoolExecutor).
  3. Collect each agent's 5 ranked picks across 4 horizons.
  4. Write every forecast row into the ADX `agentStockForecast` table.
  5. Print a consolidated summary report.

Usage:
  python -m agents.parent_agent
  python -m agents.parent_agent --date 2025-01-15
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from typing import Any

from agents.adx_client import get_all_symbols, get_price_history, ingest_forecasts, now_utc_iso
from agents.child_agents import CHILD_AGENTS, run_child_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HORIZON_KEYS = {
    "1m": "expected_return_1m",
    "3m": "expected_return_3m",
    "6m": "expected_return_6m",
    "1y": "expected_return_1y",
}
PRICE_HISTORY_DAYS = 30
MAX_WORKERS = 9  # one worker per child agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_symbols(stock_data: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Return the list of symbols present in the ADX price data."""
    return sorted(stock_data.keys())


def _build_forecast_rows(
    agent_name: str,
    picks: list[dict[str, Any]],
    report_time: str,
) -> list[dict[str, Any]]:
    """Convert a child agent's picks into ADX forecast rows."""
    rows: list[dict[str, Any]] = []
    for pick in picks:
        symbol = pick.get("symbol", "")
        rank = pick.get("rank", 0)
        for horizon, key in HORIZON_KEYS.items():
            expected_return = pick.get(key)
            if expected_return is None:
                continue
            rows.append(
                {
                    "reportTime": report_time,
                    "agentName": agent_name,
                    "symbol": symbol,
                    "horizon": horizon,
                    "expectedReturn": float(expected_return),
                    "rank": int(rank),
                }
            )
    return rows


def _print_summary(
    all_forecasts: list[dict[str, Any]],
    report_time: str,
) -> None:
    """Print a human-readable summary grouped by horizon."""
    print("\n" + "=" * 70)
    print(f"Stonks.ai Multi-Agent Forecast  |  {report_time}")
    print("=" * 70)

    for horizon in ["1m", "3m", "6m", "1y"]:
        horizon_rows = [r for r in all_forecasts if r["horizon"] == horizon]
        # Aggregate: sum expected returns across agents for each symbol
        symbol_returns: dict[str, list[float]] = {}
        for row in horizon_rows:
            symbol_returns.setdefault(row["symbol"], []).append(row["expectedReturn"])

        # Rank by mean expected return (highest first)
        ranked = sorted(
            symbol_returns.items(),
            key=lambda kv: sum(kv[1]) / len(kv[1]),
            reverse=True,
        )[:5]

        print(f"\nTop 5 picks — {horizon} horizon:")
        print(f"  {'Rank':<6} {'Symbol':<8} {'Avg Expected Return':>20}")
        print(f"  {'-'*4:<6} {'-'*6:<8} {'-'*20:>20}")
        for i, (symbol, returns) in enumerate(ranked, start=1):
            avg = sum(returns) / len(returns)
            print(f"  {i:<6} {symbol:<8} {avg:>19.2f}%")

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_parent_agent(as_of_date: date | None = None) -> list[dict[str, Any]]:
    """
    Execute the full multi-agent pipeline and return all forecast rows.

    Args:
        as_of_date: Optional date to treat as "today" for price history queries
            and the forecast reportTime.  Defaults to the actual UTC now.
            Pass a historical date to backfill forecasts.
    """
    if as_of_date is not None:
        report_time = datetime(
            as_of_date.year, as_of_date.month, as_of_date.day,
            tzinfo=timezone.utc,
        ).isoformat()
        as_of_str = as_of_date.isoformat()
    else:
        report_time = now_utc_iso()
        as_of_str = None

    # ------------------------------------------------------------------
    # 1. Pull price data from ADX
    # ------------------------------------------------------------------
    log.info("Fetching all symbols from ADX (last %d days)…", PRICE_HISTORY_DAYS)
    # An empty symbol list causes get_price_history to return all symbols;
    # we query broadly then pass the full dataset to every child agent.
    stock_data = _fetch_all_price_history(days=PRICE_HISTORY_DAYS, as_of_date=as_of_str)

    if not stock_data:
        log.error(
            "No price data returned from ADX. "
            "Run the snapshot scraper first: python -m agents.scraper --mode snapshot"
        )
        raise RuntimeError("ADX returned no price data.")

    symbols = _discover_symbols(stock_data)
    log.info("Loaded price data for %d symbols.", len(symbols))

    # ------------------------------------------------------------------
    # 2. Fan out to child agents concurrently
    # ------------------------------------------------------------------
    all_forecast_rows: list[dict[str, Any]] = []

    log.info("Running %d child agents concurrently…", len(CHILD_AGENTS))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                run_child_agent,
                agent["name"],
                agent["strategy"],
                stock_data,
            ): agent["name"]
            for agent in CHILD_AGENTS
        }

        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                picks = future.result()
                log.info("Agent '%s' returned %d picks.", agent_name, len(picks))
                rows = _build_forecast_rows(agent_name, picks, report_time)
                all_forecast_rows.extend(rows)
            except Exception as exc:
                log.error("Agent '%s' failed: %s", agent_name, exc)

    # ------------------------------------------------------------------
    # 3. Persist forecasts to ADX
    # ------------------------------------------------------------------
    if all_forecast_rows:
        log.info("Ingesting %d forecast rows into ADX…", len(all_forecast_rows))
        ingest_forecasts(all_forecast_rows)
        log.info("Forecast ingest complete.")
    else:
        log.warning("No forecast rows to ingest — all agents may have failed.")

    # ------------------------------------------------------------------
    # 4. Print summary
    # ------------------------------------------------------------------
    _print_summary(all_forecast_rows, report_time)

    return all_forecast_rows


def _fetch_all_price_history(
    days: int, as_of_date: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    """
    Fetch price history for all symbols present in ADX for the given window.

    Args:
        days: Number of days to look back.
        as_of_date: Optional ISO-8601 date string.  When provided, the window
            ends at midnight of this date rather than "now".
    """
    if days < 1:
        raise ValueError(f"days must be a positive integer, got {days!r}")
    symbols = get_all_symbols(days=days, as_of_date=as_of_date)
    if not symbols:
        return {}
    return get_price_history(symbols, days=days, as_of_date=as_of_date)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stonks.ai parent (orchestrator) agent")
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help=(
            "Treat this ISO-8601 date (YYYY-MM-DD) as 'today' when generating "
            "forecasts.  Price history is bounded to the 30-day window ending on "
            "this date and reportTime is set to midnight UTC of this date. "
            "Defaults to the actual current UTC time. Useful for backfilling."
        ),
    )
    args = parser.parse_args()
    run_parent_agent(as_of_date=args.date)
