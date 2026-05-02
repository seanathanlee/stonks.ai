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
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.ingest import (
    IngestionProperties,
    QueuedIngestClient,
    DataFormat,
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
dailyStockPrice
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


def now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
