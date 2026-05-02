"""
ADX (Azure Data Explorer / Kusto) data access layer for Stonks.ai.

Provides helpers to:
  - Query historical daily stock prices from the `dailyStockPrice` table.
  - Ingest daily stock price rows into `dailyStockPrice`.
  - Ingest agent forecast rows into `agentStockForecast`.

Configuration (environment variables):
  ADX_CLUSTER_URI  – e.g. https://stonksaiadx.eastus.kusto.windows.net
  ADX_DATABASE     – e.g. stonksai
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from typing import Any

from azure.identity import DefaultAzureCredential
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder, DataFormat
from azure.kusto.ingest import (
    IngestionProperties,
    QueuedIngestClient,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_kusto_client: KustoClient | None = None
_ingest_client: QueuedIngestClient | None = None


def _cluster_uri() -> str:
    uri = os.environ.get("ADX_CLUSTER_URI", "")
    if not uri:
        raise EnvironmentError("ADX_CLUSTER_URI environment variable is not set.")
    return uri


def _database() -> str:
    return os.environ.get("ADX_DATABASE", "stonksai")


def _get_kusto_client() -> KustoClient:
    global _kusto_client
    if _kusto_client is None:
        credential = DefaultAzureCredential()
        kcsb = KustoConnectionStringBuilder.with_azure_token_credential(
            _cluster_uri(), credential
        )
        _kusto_client = KustoClient(kcsb)
    return _kusto_client


def _get_ingest_client() -> QueuedIngestClient:
    global _ingest_client
    if _ingest_client is None:
        # Ingest endpoint uses a different subdomain
        ingest_uri = _cluster_uri().replace("https://", "https://ingest-", 1)
        credential = DefaultAzureCredential()
        kcsb = KustoConnectionStringBuilder.with_azure_token_credential(
            ingest_uri, credential
        )
        _ingest_client = QueuedIngestClient(kcsb)
    return _ingest_client


def _rows_to_json_stream(rows: list[dict[str, Any]]) -> io.BytesIO:
    """Serialize a list of dicts to newline-delimited JSON bytes."""
    lines = "\n".join(json.dumps(row) for row in rows)
    return io.BytesIO(lines.encode("utf-8"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_all_symbols(days: int = 30) -> list[str]:
    """
    Return the distinct set of symbols that have price data in ADX
    for the last *days* days.
    """
    if days < 1:
        raise ValueError(f"days must be a positive integer, got {days!r}")
    client = _get_kusto_client()
    query = f"dailyStockPriceMV | where priceDate >= ago({int(days)}d) | summarize by symbol"
    response = client.execute(_database(), query)
    return [row["symbol"] for row in response.primary_results[0]]


def get_price_history(
    symbols: list[str], days: int = 30
) -> dict[str, list[dict[str, Any]]]:
    """
    Query the last *days* of daily closing prices for the given symbols.

    Returns a dict keyed by symbol, each value being a list of records:
        {"date": "2024-01-15", "price": 182.50}
    sorted chronologically (oldest first).
    """
    if not symbols:
        return {}

    symbol_list = ", ".join(f'"{s}"' for s in symbols)
    query = f"""
dailyStockPriceMV
| where priceDate >= ago({days}d)
| where symbol in ({symbol_list})
| project priceDate, symbol, price
| order by priceDate asc
"""
    client = _get_kusto_client()
    response = client.execute(_database(), query)

    result: dict[str, list[dict[str, Any]]] = {s: [] for s in symbols}
    for row in response.primary_results[0]:
        symbol = row["symbol"]
        result.setdefault(symbol, []).append(
            {
                "date": row["priceDate"].strftime("%Y-%m-%d")
                if hasattr(row["priceDate"], "strftime")
                else str(row["priceDate"]),
                "price": float(row["price"]),
            }
        )
    return result


def ingest_daily_prices(rows: list[dict[str, Any]]) -> None:
    """
    Bulk-ingest rows into the `dailyStockPrice` table.

    Each row must contain:
        reportTime  – ISO-8601 datetime string (when the row was written)
        symbol      – stock ticker string
        price       – closing price (float)
        priceDate   – ISO-8601 date string (trading date)
    """
    if not rows:
        return

    props = IngestionProperties(
        database=_database(),
        table="dailyStockPrice",
        data_format=DataFormat.MULTIJSON,
    )
    stream = _rows_to_json_stream(rows)
    _get_ingest_client().ingest_from_stream(stream, ingestion_properties=props)


def ingest_forecasts(rows: list[dict[str, Any]]) -> None:
    """
    Bulk-ingest rows into the `agentStockForecast` table.

    Each row must contain:
        reportTime     – ISO-8601 datetime string
        agentName      – child agent name string
        symbol         – stock ticker string
        horizon        – one of "1m", "3m", "6m", "1y"
        expectedReturn – forecasted percentage return (float)
        rank           – integer rank (1 = best) within agent+horizon
    """
    if not rows:
        return

    props = IngestionProperties(
        database=_database(),
        table="agentStockForecast",
        data_format=DataFormat.MULTIJSON,
    )
    stream = _rows_to_json_stream(rows)
    _get_ingest_client().ingest_from_stream(stream, ingestion_properties=props)


def get_latest_forecasts(
    horizon: str = "1m",
    symbol: str | None = None,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Return the top *top_n* stock picks from the most recent agent run.

    Aggregates expected returns across all child agents and ranks by the mean.
    Each returned dict contains:
        symbol        – stock ticker
        avgReturn     – average expected % return across agents
        agentCount    – number of agents that included this symbol
        horizon       – the horizon used for filtering
    """
    if horizon not in ("1m", "3m", "6m", "1y"):
        raise ValueError(f"Invalid horizon: {horizon!r}. Must be one of 1m, 3m, 6m, 1y.")
    if top_n < 1:
        raise ValueError(f"top_n must be a positive integer, got {top_n!r}")

    symbol_filter = f'| where symbol == "{symbol}"' if symbol else ""
    query = f"""
agentStockForecastMV
| where reportTime >= ago(7d)
| where horizon == "{horizon}"
{symbol_filter}
| summarize avgReturn = round(avg(expectedReturn), 2), agentCount = dcount(agentName) by symbol
| order by avgReturn desc
| take {int(top_n)}
"""
    client = _get_kusto_client()
    response = client.execute(_database(), query)
    return [
        {
            "symbol": row["symbol"],
            "avgReturn": float(row["avgReturn"]),
            "agentCount": int(row["agentCount"]),
            "horizon": horizon,
        }
        for row in response.primary_results[0]
    ]


def get_agent_comparison(
    symbol: str,
    horizon: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return per-agent expected returns for a specific symbol from the most recent run.

    Each returned dict contains:
        agentName      – child agent name
        horizon        – investment horizon
        expectedReturn – forecasted percentage return
    Rows are sorted by horizon then descending expected return.
    """
    horizon_filter = f'| where horizon == "{horizon}"' if horizon else ""
    query = f"""
agentStockForecastMV
| where reportTime >= ago(7d)
| where symbol == "{symbol}"
{horizon_filter}
| summarize expectedReturn = round(avg(expectedReturn), 2) by agentName, horizon
| order by horizon asc, expectedReturn desc
"""
    client = _get_kusto_client()
    response = client.execute(_database(), query)
    return [
        {
            "agentName": row["agentName"],
            "horizon": row["horizon"],
            "expectedReturn": float(row["expectedReturn"]),
        }
        for row in response.primary_results[0]
    ]


def now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
