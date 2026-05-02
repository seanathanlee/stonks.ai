"""
Stonks.ai Scraper
Fetches NASDAQ stock symbols and their daily closing prices, then ingests
the data into the Azure Data Explorer `dailyStockPrice` table.

Usage:
  python -m agents.scraper --mode daily     # today's price for all symbols
  python -m agents.scraper --mode snapshot  # last 30 days for all symbols
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, timezone, datetime

import requests
from bs4 import BeautifulSoup

from agents.adx_client import ingest_daily_prices, now_utc_iso

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOL_LIST_URL = "https://stockanalysis.com/list/nasdaq-stocks/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; stonks-scraper/1.0; +https://github.com/seanathanlee/stonks.ai)"
    )
}
REQUEST_DELAY_SECONDS = 0.5
SNAPSHOT_DAYS = 30

# ---------------------------------------------------------------------------
# Symbol discovery
# ---------------------------------------------------------------------------


def fetch_nasdaq_symbols() -> list[str]:
    """Scrape the NASDAQ stock list and return a list of ticker symbols."""
    log.info("Fetching NASDAQ symbol list from %s", SYMBOL_LIST_URL)
    resp = requests.get(SYMBOL_LIST_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    symbols: list[str] = []
    # stockanalysis.com renders symbols in a <table>; find all table cells
    # whose text looks like a ticker (2–5 uppercase letters)
    for cell in soup.select("table td:first-child a, table td a"):
        text = cell.get_text(strip=True)
        # Basic filter: all-uppercase, 1-5 chars, alphabetic only
        if text and text.isupper() and text.isalpha() and 1 <= len(text) <= 5:
            symbols.append(text)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    log.info("Discovered %d NASDAQ symbols", len(unique))
    return unique


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------


def fetch_current_price(symbol: str) -> float | None:
    """
    Fetch the latest price for *symbol* from Yahoo Finance's quote endpoint.
    Returns the price as a float, or None if the request fails.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        meta = data["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice") or meta.get("previousClose")
        return float(price) if price is not None else None
    except Exception as exc:
        log.warning("Could not fetch price for %s: %s", symbol, exc)
        return None


def fetch_price_history(
    symbol: str, days: int = SNAPSHOT_DAYS
) -> list[tuple[date, float]]:
    """
    Fetch the last *days* of daily closing prices for *symbol* from Yahoo Finance.
    Returns a list of (trade_date, close_price) tuples, oldest first.
    """
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = end_ts - days * 86400
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=1d&period1={start_ts}&period2={end_ts}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
        pairs: list[tuple[date, float]] = []
        for ts, price in zip(timestamps, closes):
            if price is None:
                continue
            trade_date = datetime.fromtimestamp(ts, tz=timezone.utc).date()
            pairs.append((trade_date, float(price)))
        return pairs
    except Exception as exc:
        log.warning("Could not fetch history for %s: %s", symbol, exc)
        return []


# ---------------------------------------------------------------------------
# Scraper modes
# ---------------------------------------------------------------------------


def run_daily(symbols: list[str]) -> None:
    """Fetch today's closing price for every symbol and ingest into ADX."""
    report_time = now_utc_iso()
    today = datetime.now(timezone.utc).date().isoformat()
    rows: list[dict] = []

    for i, symbol in enumerate(symbols, start=1):
        price = fetch_current_price(symbol)
        if price is not None:
            rows.append(
                {
                    "reportTime": report_time,
                    "symbol": symbol,
                    "price": price,
                    "priceDate": today,
                }
            )
        if i % 100 == 0:
            log.info("Progress: %d / %d symbols processed", i, len(symbols))
        time.sleep(REQUEST_DELAY_SECONDS)

    log.info("Ingesting %d price rows into ADX (daily mode)", len(rows))
    ingest_daily_prices(rows)
    log.info("Daily ingest complete.")


def run_snapshot(symbols: list[str], days: int = SNAPSHOT_DAYS) -> None:
    """Fetch the last *days* of prices for every symbol and ingest into ADX."""
    report_time = now_utc_iso()
    rows: list[dict] = []

    for i, symbol in enumerate(symbols, start=1):
        history = fetch_price_history(symbol, days=days)
        for trade_date, price in history:
            rows.append(
                {
                    "reportTime": report_time,
                    "symbol": symbol,
                    "price": price,
                    "priceDate": trade_date.isoformat(),
                }
            )
        if i % 50 == 0:
            log.info("Progress: %d / %d symbols processed", i, len(symbols))
        time.sleep(REQUEST_DELAY_SECONDS)

    log.info("Ingesting %d price rows into ADX (snapshot mode)", len(rows))
    ingest_daily_prices(rows)
    log.info("Snapshot ingest complete.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Stonks.ai stock price scraper")
    parser.add_argument(
        "--mode",
        choices=["daily", "snapshot"],
        required=True,
        help="'daily' ingests today's prices; 'snapshot' ingests the last 30 days.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=SNAPSHOT_DAYS,
        help="Number of days to back-fill in snapshot mode (default: 30).",
    )
    args = parser.parse_args()

    symbols = fetch_nasdaq_symbols()
    if not symbols:
        log.error("No symbols discovered — aborting.")
        raise SystemExit(1)

    if args.mode == "daily":
        run_daily(symbols)
    else:
        run_snapshot(symbols, days=args.days)


if __name__ == "__main__":
    main()
