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


_ISO_DATE_RE = __import__("re").compile(r"^\d{4}-\d{2}-\d{2}$")
_HORIZON_RE = __import__("re").compile(r"^[a-z0-9]{1,4}$")


def _validate_iso_date(value: str, name: str) -> None:
    """Raise ValueError if *value* is not a plain YYYY-MM-DD date string."""
    if not _ISO_DATE_RE.match(value):
        raise ValueError(f"{name} must be an ISO-8601 date (YYYY-MM-DD), got {value!r}")


def get_forecasts_from_date(
    target_date: str,
    horizon: str = "1m",
) -> list[dict[str, Any]]:
    """
    Return all agent forecast rows whose reportTime falls on *target_date* (UTC).

    *target_date* must be an ISO-8601 date string, e.g. "2026-04-02".
    Returns a list of dicts with keys:
        reportTime, agentName, symbol, horizon, expectedReturn, rank
    """
    _validate_iso_date(target_date, "target_date")
    if not _HORIZON_RE.match(horizon):
        raise ValueError(f"horizon must be alphanumeric (max 4 chars), got {horizon!r}")
    query = f"""
agentStockForecast
| where reportTime >= datetime({target_date}T00:00:00Z)
  and reportTime  <  datetime({target_date}T00:00:00Z) + 1d
| where horizon == "{horizon}"
| project reportTime, agentName, symbol, horizon, expectedReturn, rank
"""
    client = _get_kusto_client()
    response = client.execute(_database(), query)
    results = []
    for row in response.primary_results[0]:
        results.append(
            {
                "reportTime": row["reportTime"].isoformat()
                if hasattr(row["reportTime"], "isoformat")
                else str(row["reportTime"]),
                "agentName": row["agentName"],
                "symbol": row["symbol"],
                "horizon": row["horizon"],
                "expectedReturn": float(row["expectedReturn"]),
                "rank": int(row["rank"]),
            }
        )
    return results


def get_price_range(
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, list[dict[str, Any]]]:
    """
    Query daily closing prices for *symbols* between *start_date* and *end_date*
    (inclusive, ISO-8601 date strings).

    Returns a dict keyed by symbol, each value being a list of records:
        {"date": "2024-01-15", "price": 182.50}
    sorted chronologically (oldest first).
    """
    if not symbols:
        return {}

    _validate_iso_date(start_date, "start_date")
    _validate_iso_date(end_date, "end_date")

    symbol_list = ", ".join(f'"{s}"' for s in symbols)
    query = f"""
dailyStockPriceMV
| where priceDate >= datetime({start_date}T00:00:00Z)
  and priceDate <= datetime({end_date}T23:59:59Z)
| where symbol in ({symbol_list})
| project priceDate, symbol, price
| order by priceDate asc
"""
    client = _get_kusto_client()
    response = client.execute(_database(), query)

    result: dict[str, list[dict[str, Any]]] = {s: [] for s in symbols}
    for row in response.primary_results[0]:
        symbol = row["symbol"]
        result[symbol].append(
            {
                "date": row["priceDate"].strftime("%Y-%m-%d")
                if hasattr(row["priceDate"], "strftime")
                else str(row["priceDate"]),
                "price": float(row["price"]),
            }
        )
    return result


def ingest_evaluations(rows: list[dict[str, Any]]) -> None:
    """
    Bulk-ingest rows into the `agentStockEvaluation` table.

    Each row must contain:
        symbol              – stock ticker string
        forecastReturn      – forecasted % return (float)
        actualReturn        – realized % return (float)
        forecastRank        – rank assigned by the forecasting agent (int)
        actualRank          – rank based on actual returns (int)
        accuracyScore       – accuracy metric (float)
        agentName           – name of the forecasting agent (string)
        forecastReportTime  – ISO-8601 datetime of the original forecast
        reportTime          – ISO-8601 datetime of this evaluation run
        runId               – unique ID for this evaluation run (string)
        horizon             – forecast horizon, e.g. "1m" (string)
    """
    if not rows:
        return

    props = IngestionProperties(
        database=_database(),
        table="agentStockEvaluation",
        data_format=DataFormat.MULTIJSON,
    )
    stream = _rows_to_json_stream(rows)
    _get_ingest_client().ingest_from_stream(stream, ingestion_properties=props)


def get_agent_evaluations(
    horizon: str = "1m",
    agent_name: str | None = None,
    days: int = 30,
) -> list[dict[str, Any]]:
    """
    Return per-agent accuracy evaluation metrics from the `agentStockEvaluation` table.

    The accuracy score is a lower-is-better error metric: the average of the
    absolute return error and absolute rank error for each prediction.

    Each returned dict contains:
        agentName           – child agent name
        avgAccuracyScore    – mean accuracy score (lower = more accurate)
        runCount            – number of distinct evaluation runs included
        avgForecastReturn   – mean forecasted % return across evaluated picks
        avgActualReturn     – mean realized % return across evaluated picks
        horizon             – the horizon used for filtering
    """
    if days < 1:
        raise ValueError(f"days must be a positive integer, got {days!r}")
    if horizon not in ("1m", "3m", "6m", "1y"):
        raise ValueError(f"Invalid horizon: {horizon!r}. Must be one of 1m, 3m, 6m, 1y.")

    agent_filter = f'| where agentName == "{agent_name}"' if agent_name else ""
    query = f"""
agentStockEvaluation
| where reportTime >= ago({int(days)}d)
| where horizon == "{horizon}"
{agent_filter}
| summarize avgAccuracyScore = round(avg(accuracyScore), 4),
            runCount = dcount(runId),
            avgForecastReturn = round(avg(forecastReturn), 2),
            avgActualReturn = round(avg(actualReturn), 2) by agentName
| order by avgAccuracyScore asc
"""
    client = _get_kusto_client()
    response = client.execute(_database(), query)
    return [
        {
            "agentName": row["agentName"],
            "avgAccuracyScore": float(row["avgAccuracyScore"]),
            "runCount": int(row["runCount"]),
            "avgForecastReturn": float(row["avgForecastReturn"]),
            "avgActualReturn": float(row["avgActualReturn"]),
            "horizon": horizon,
        }
        for row in response.primary_results[0]
    ]


def now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
