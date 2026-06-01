from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class Holding:
    symbol: str
    quantity: float
    market_price: float | None = None
    market_value: float | None = None


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    holdings: list[Holding]


@dataclass(frozen=True)
class TradeOrder:
    side: str
    symbol: str
    quantity: float | None = None
    notional: float | None = None
    client_order_id: str | None = None


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    status: str
    symbol: str
    side: str
    requested_quantity: float | None = None
    requested_notional: float | None = None
    filled_quantity: float | None = None
    filled_notional: float | None = None


class BrokerClient(ABC):
    @abstractmethod
    def get_account_snapshot(self) -> AccountSnapshot:
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, order: TradeOrder) -> OrderResult:
        raise NotImplementedError


class PaperBrokerClient(BrokerClient):
    def __init__(self, cash: float, holdings: list[Holding] | None = None):
        self._cash = float(cash)
        self._holdings = holdings or []

    def get_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(cash=self._cash, holdings=list(self._holdings))

    def submit_order(self, order: TradeOrder) -> OrderResult:
        order_id = order.client_order_id or f"paper-{order.side}-{order.symbol}"
        return OrderResult(
            order_id=order_id,
            status="accepted",
            symbol=order.symbol,
            side=order.side,
            requested_quantity=order.quantity,
            requested_notional=order.notional,
            filled_quantity=order.quantity,
            filled_notional=order.notional,
        )


class HttpBrokerClient(BrokerClient):
    """Generic HTTP broker adapter suitable for Robinhood-compatible endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        account_id: str,
        token: str,
        account_path_template: str = "/accounts/{account_id}",
        holdings_path_template: str = "/accounts/{account_id}/holdings",
        orders_path_template: str = "/accounts/{account_id}/orders",
        timeout_seconds: int = 30,
        cash_fields: tuple[str, ...] = (
            "cash",
            "buying_power",
            "available_cash",
            "withdrawable_cash",
        ),
    ):
        self._base_url = base_url.rstrip("/")
        self._account_id = account_id
        self._account_path_template = account_path_template
        self._holdings_path_template = holdings_path_template
        self._orders_path_template = orders_path_template
        self._timeout_seconds = timeout_seconds
        self._cash_fields = cash_fields
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def _url(self, template: str) -> str:
        return f"{self._base_url}{template.format(account_id=self._account_id)}"

    def _extract_cash(self, payload: dict[str, Any]) -> float:
        for key in self._cash_fields:
            if key in payload and payload[key] is not None:
                return float(payload[key])
        raise ValueError("Account payload missing cash/buying power fields.")

    def _extract_holdings(self, payload: Any) -> list[Holding]:
        rows = payload.get("holdings", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("Holdings response must be a list or {'holdings': [...]} object.")
        holdings: list[Holding] = []
        for item in rows:
            symbol = str(item.get("symbol", "")).upper()
            quantity = float(item.get("quantity", 0.0))
            if not symbol or quantity <= 0:
                continue
            market_price = (
                float(item["market_price"]) if item.get("market_price") is not None else None
            )
            market_value = (
                float(item["market_value"]) if item.get("market_value") is not None else None
            )
            holdings.append(
                Holding(
                    symbol=symbol,
                    quantity=quantity,
                    market_price=market_price,
                    market_value=market_value,
                )
            )
        return holdings

    def get_account_snapshot(self) -> AccountSnapshot:
        try:
            account_resp = self._session.get(
                self._url(self._account_path_template),
                timeout=self._timeout_seconds,
            )
            account_resp.raise_for_status()
            account_payload = account_resp.json()
        except Exception as exc:
            raise RuntimeError("Failed to fetch broker account snapshot.") from exc

        try:
            holdings_resp = self._session.get(
                self._url(self._holdings_path_template),
                timeout=self._timeout_seconds,
            )
            holdings_resp.raise_for_status()
            holdings_payload = holdings_resp.json()
        except Exception as exc:
            raise RuntimeError("Failed to fetch broker holdings.") from exc

        return AccountSnapshot(
            cash=self._extract_cash(account_payload),
            holdings=self._extract_holdings(holdings_payload),
        )

    def submit_order(self, order: TradeOrder) -> OrderResult:
        payload: dict[str, Any] = {
            "side": order.side,
            "symbol": order.symbol,
            "type": "market",
        }
        if order.quantity is not None:
            payload["quantity"] = order.quantity
        if order.notional is not None:
            payload["notional"] = order.notional
        if order.client_order_id:
            payload["client_order_id"] = order.client_order_id

        resp = self._session.post(
            self._url(self._orders_path_template),
            json=payload,
            timeout=self._timeout_seconds,
        )
        resp.raise_for_status()
        body = resp.json() if resp.content else {}

        return OrderResult(
            order_id=str(body.get("id", body.get("order_id", order.client_order_id or "unknown"))),
            status=str(body.get("status", "accepted")),
            symbol=order.symbol,
            side=order.side,
            requested_quantity=order.quantity,
            requested_notional=order.notional,
            filled_quantity=float(body["filled_quantity"]) if body.get("filled_quantity") is not None else None,
            filled_notional=float(body["filled_notional"]) if body.get("filled_notional") is not None else None,
        )
