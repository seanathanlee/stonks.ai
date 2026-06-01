from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import adx_client
from agents.broker import (
    AccountSnapshot,
    BrokerClient,
    Holding,
    HttpBrokerClient,
    OrderResult,
    PaperBrokerClient,
    TradeOrder,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_TOP_N = 5
DEFAULT_HORIZON = "1m"


class RebalanceError(RuntimeError):
    pass


class AuditLog:
    def __init__(self, path: str | None = None):
        self.path = Path(path) if path else None
        self.events: list[dict[str, Any]] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **payload,
        }
        self.events.append(row)
        line = json.dumps(row, sort_keys=True)
        log.info(line)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def summary_markdown(self) -> str:
        selected = next((e for e in self.events if e["event"] == "selection"), None)
        plan = next((e for e in self.events if e["event"] == "plan"), None)
        executed = [e for e in self.events if e["event"] == "order_executed"]
        failed = [e for e in self.events if e["event"] == "order_failed"]

        lines = [
            "## Agent Rebalance Summary",
            f"- Status: {'FAILED' if failed else 'SUCCESS'}",
            f"- Top picks: {', '.join(selected['symbols']) if selected else 'n/a'}",
            f"- Orders executed: {len(executed)}",
            f"- Orders failed: {len(failed)}",
        ]
        if plan:
            lines.extend(
                [
                    f"- Estimated liquidation value: ${plan['estimated_liquidation_value']:.2f}",
                    f"- Buy notional per symbol: ${plan['buy_notional_per_symbol']:.2f}",
                    f"- Unallocated cash: ${plan['unallocated_cash']:.2f}",
                ]
            )
        return "\n".join(lines) + "\n"


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def select_top_symbols(*, horizon: str, top_n: int) -> list[str]:
    forecasts = adx_client.get_latest_forecasts(horizon=horizon, top_n=top_n)
    symbols = []
    seen: set[str] = set()
    for row in forecasts:
        symbol = _normalize_symbol(str(row.get("symbol", "")))
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)

    if len(symbols) < top_n:
        raise RebalanceError(
            f"Expected at least {top_n} unique picks for horizon {horizon}, got {len(symbols)}."
        )
    return symbols[:top_n]


def _holding_value(holding: Holding) -> float:
    if holding.market_value is not None:
        return float(holding.market_value)
    if holding.market_price is None:
        raise RebalanceError(
            f"Holding {holding.symbol} missing market_price/market_value; cannot size rebalance."
        )
    return float(holding.quantity) * float(holding.market_price)


def build_rebalance_plan(
    *,
    snapshot: AccountSnapshot,
    target_symbols: list[str],
    min_cash_threshold: float,
    max_order_notional: float,
) -> dict[str, Any]:
    if len(target_symbols) < DEFAULT_TOP_N:
        raise RebalanceError("Need at least 5 target symbols for rebalance.")
    if max_order_notional <= 0:
        raise RebalanceError("max_order_notional must be > 0.")

    estimated_holdings_value = sum(_holding_value(h) for h in snapshot.holdings)
    estimated_liquidation_value = float(snapshot.cash) + estimated_holdings_value

    if estimated_liquidation_value < min_cash_threshold:
        raise RebalanceError(
            f"Estimated liquidation value ${estimated_liquidation_value:.2f} is below threshold "
            f"${min_cash_threshold:.2f}."
        )

    buy_notional_per_symbol = min(
        estimated_liquidation_value / len(target_symbols),
        max_order_notional,
    )

    sell_orders = [
        {
            "side": "sell",
            "symbol": h.symbol,
            "quantity": float(h.quantity),
        }
        for h in snapshot.holdings
        if h.quantity > 0
    ]

    buy_orders = [
        {
            "side": "buy",
            "symbol": symbol,
            "notional": float(round(buy_notional_per_symbol, 2)),
        }
        for symbol in target_symbols
    ]

    planned_buy_notional = sum(order["notional"] for order in buy_orders)

    return {
        "estimated_holdings_value": estimated_holdings_value,
        "estimated_liquidation_value": estimated_liquidation_value,
        "buy_notional_per_symbol": float(round(buy_notional_per_symbol, 2)),
        "planned_buy_notional": float(round(planned_buy_notional, 2)),
        "unallocated_cash": float(round(estimated_liquidation_value - planned_buy_notional, 2)),
        "sell_orders": sell_orders,
        "buy_orders": buy_orders,
    }


def _submit_order_with_retry(
    *,
    broker: BrokerClient,
    order: TradeOrder,
    max_retries: int,
) -> OrderResult:
    last_exc: Exception | None = None
    for _ in range(max_retries):
        try:
            return broker.submit_order(order)
        except Exception as exc:
            last_exc = exc
    raise RebalanceError(f"Order failed after {max_retries} attempts: {order}") from last_exc


def execute_rebalance_plan(
    *,
    broker: BrokerClient,
    plan: dict[str, Any],
    audit: AuditLog,
    dry_run: bool,
    run_id: str,
    retries: int,
) -> list[OrderResult]:
    order_results: list[OrderResult] = []
    serial = 0

    for payload in (plan["sell_orders"] + plan["buy_orders"]):
        serial += 1
        order = TradeOrder(
            side=str(payload["side"]),
            symbol=_normalize_symbol(str(payload["symbol"])),
            quantity=float(payload["quantity"]) if payload.get("quantity") is not None else None,
            notional=float(payload["notional"]) if payload.get("notional") is not None else None,
            client_order_id=f"{run_id}-{serial}-{payload['side']}-{payload['symbol']}",
        )

        if dry_run:
            result = OrderResult(
                order_id=f"dry-run-{order.client_order_id}",
                status="simulated",
                symbol=order.symbol,
                side=order.side,
                requested_quantity=order.quantity,
                requested_notional=order.notional,
                filled_quantity=order.quantity,
                filled_notional=order.notional,
            )
        else:
            try:
                result = _submit_order_with_retry(broker=broker, order=order, max_retries=retries)
            except Exception as exc:
                audit.emit(
                    "order_failed",
                    {
                        "run_id": run_id,
                        "dry_run": dry_run,
                        "order": asdict(order),
                        "error": str(exc),
                    },
                )
                raise

        order_results.append(result)
        audit.emit(
            "order_executed",
            {
                "run_id": run_id,
                "dry_run": dry_run,
                "result": asdict(result),
            },
        )

    return order_results


def _parse_holdings_json(value: str | None) -> list[Holding]:
    if not value:
        return []
    raw = json.loads(value)
    if not isinstance(raw, list):
        raise RebalanceError("PAPER_HOLDINGS_JSON must be a JSON array.")
    holdings: list[Holding] = []
    for item in raw:
        holdings.append(
            Holding(
                symbol=_normalize_symbol(str(item.get("symbol", ""))),
                quantity=float(item.get("quantity", 0.0)),
                market_price=float(item["market_price"]) if item.get("market_price") is not None else None,
                market_value=float(item["market_value"]) if item.get("market_value") is not None else None,
            )
        )
    return [h for h in holdings if h.symbol and h.quantity > 0]


def build_broker_from_env(provider: str) -> BrokerClient:
    provider = provider.strip().lower()
    if provider == "paper":
        cash = float(os.environ.get("PAPER_CASH", "0"))
        holdings = _parse_holdings_json(os.environ.get("PAPER_HOLDINGS_JSON"))
        return PaperBrokerClient(cash=cash, holdings=holdings)

    if provider in {"robinhood", "http"}:
        base_url = os.environ.get("BROKER_API_BASE_URL")
        account_id = os.environ.get("BROKER_ACCOUNT_ID")
        token = os.environ.get("BROKER_API_TOKEN")
        if not base_url or not account_id or not token:
            raise RebalanceError(
                "BROKER_API_BASE_URL, BROKER_ACCOUNT_ID, and BROKER_API_TOKEN are required "
                f"for provider '{provider}'."
            )
        return HttpBrokerClient(
            base_url=base_url,
            account_id=account_id,
            token=token,
            account_path_template=os.environ.get("BROKER_ACCOUNT_PATH", "/accounts/{account_id}"),
            holdings_path_template=os.environ.get(
                "BROKER_HOLDINGS_PATH", "/accounts/{account_id}/holdings"
            ),
            orders_path_template=os.environ.get("BROKER_ORDERS_PATH", "/accounts/{account_id}/orders"),
            timeout_seconds=int(os.environ.get("BROKER_HTTP_TIMEOUT", "30")),
        )

    raise RebalanceError(f"Unsupported broker provider: {provider}")


def run_rebalance(
    *,
    broker_provider: str,
    horizon: str,
    top_n: int,
    min_cash_threshold: float,
    max_order_notional: float,
    dry_run: bool,
    retries: int,
    audit_log_path: str | None,
    run_id: str,
) -> dict[str, Any]:
    audit = AuditLog(audit_log_path)
    broker = build_broker_from_env(broker_provider)

    symbols = select_top_symbols(horizon=horizon, top_n=top_n)
    audit.emit(
        "selection",
        {
            "run_id": run_id,
            "horizon": horizon,
            "top_n": top_n,
            "symbols": symbols,
        },
    )

    snapshot = broker.get_account_snapshot()
    audit.emit(
        "account_snapshot",
        {
            "run_id": run_id,
            "cash": snapshot.cash,
            "holdings": [asdict(h) for h in snapshot.holdings],
        },
    )

    plan = build_rebalance_plan(
        snapshot=snapshot,
        target_symbols=symbols,
        min_cash_threshold=min_cash_threshold,
        max_order_notional=max_order_notional,
    )
    audit.emit(
        "plan",
        {
            "run_id": run_id,
            **plan,
        },
    )

    results = execute_rebalance_plan(
        broker=broker,
        plan=plan,
        audit=audit,
        dry_run=dry_run,
        run_id=run_id,
        retries=retries,
    )

    summary = audit.summary_markdown()
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(summary)

    return {
        "run_id": run_id,
        "selected_symbols": symbols,
        "plan": plan,
        "orders": [asdict(r) for r in results],
        "audit_events": audit.events,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a top-picks portfolio rebalance.")
    parser.add_argument("--broker-provider", default=os.environ.get("BROKER_PROVIDER", "paper"))
    parser.add_argument("--horizon", default=os.environ.get("REBALANCE_HORIZON", DEFAULT_HORIZON))
    parser.add_argument("--top-n", type=int, default=int(os.environ.get("REBALANCE_TOP_N", str(DEFAULT_TOP_N))))
    parser.add_argument(
        "--min-cash-threshold",
        type=float,
        default=float(os.environ.get("REBALANCE_MIN_CASH_THRESHOLD", "100")),
    )
    parser.add_argument(
        "--max-order-notional",
        type=float,
        default=float(os.environ.get("REBALANCE_MAX_ORDER_NOTIONAL", "100000")),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=int(os.environ.get("REBALANCE_ORDER_RETRIES", "3")),
    )
    parser.add_argument(
        "--audit-log-path",
        default=os.environ.get("REBALANCE_AUDIT_LOG_PATH", "/tmp/rebalance_audit.jsonl"),
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("REBALANCE_RUN_ID")
        or os.environ.get("GITHUB_RUN_ID")
        or str(uuid.uuid4()),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=str(os.environ.get("REBALANCE_DRY_RUN", "true")).lower() == "true",
    )
    parser.add_argument("--no-dry-run", action="store_false", dest="dry_run")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_rebalance(
        broker_provider=args.broker_provider,
        horizon=args.horizon,
        top_n=args.top_n,
        min_cash_threshold=args.min_cash_threshold,
        max_order_notional=args.max_order_notional,
        dry_run=args.dry_run,
        retries=args.retries,
        audit_log_path=args.audit_log_path,
        run_id=args.run_id,
    )
