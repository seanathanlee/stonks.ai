from __future__ import annotations

import pytest

from agents.broker import AccountSnapshot, Holding
from agents.rebalance_agent import RebalanceError, build_rebalance_plan, select_top_symbols


class TestSelectTopSymbols:
    def test_selects_unique_uppercase_symbols(self, monkeypatch):
        monkeypatch.setattr(
            "agents.rebalance_agent.adx_client.get_latest_forecasts",
            lambda horizon, top_n: [
                {"symbol": "aapl"},
                {"symbol": "MSFT"},
                {"symbol": "aapl"},
                {"symbol": "NVDA"},
                {"symbol": "AMZN"},
                {"symbol": "META"},
            ],
        )
        assert select_top_symbols(horizon="1m", top_n=5) == ["AAPL", "MSFT", "NVDA", "AMZN", "META"]

    def test_fails_when_fewer_than_requested(self, monkeypatch):
        monkeypatch.setattr(
            "agents.rebalance_agent.adx_client.get_latest_forecasts",
            lambda horizon, top_n: [{"symbol": "AAPL"}, {"symbol": "MSFT"}],
        )
        with pytest.raises(RebalanceError):
            select_top_symbols(horizon="1m", top_n=5)


class TestBuildRebalancePlan:
    def test_liquidates_all_holdings_and_buys_equal_notional(self):
        snapshot = AccountSnapshot(
            cash=100.0,
            holdings=[
                Holding(symbol="TSLA", quantity=2, market_price=50.0),
                Holding(symbol="NFLX", quantity=1, market_value=80.0),
            ],
        )

        plan = build_rebalance_plan(
            snapshot=snapshot,
            target_symbols=["AAPL", "MSFT", "NVDA", "AMZN", "META"],
            min_cash_threshold=50.0,
            max_order_notional=1000.0,
        )

        assert len(plan["sell_orders"]) == 2
        assert plan["estimated_liquidation_value"] == pytest.approx(280.0)
        assert plan["buy_notional_per_symbol"] == pytest.approx(56.0)
        assert len(plan["buy_orders"]) == 5
        assert all(order["notional"] == pytest.approx(56.0) for order in plan["buy_orders"])

    def test_caps_buy_notional_with_guardrail(self):
        snapshot = AccountSnapshot(
            cash=1000.0,
            holdings=[Holding(symbol="TSLA", quantity=10, market_price=100.0)],
        )

        plan = build_rebalance_plan(
            snapshot=snapshot,
            target_symbols=["AAPL", "MSFT", "NVDA", "AMZN", "META"],
            min_cash_threshold=100.0,
            max_order_notional=200.0,
        )

        assert plan["buy_notional_per_symbol"] == pytest.approx(200.0)
        assert plan["planned_buy_notional"] == pytest.approx(1000.0)
        assert plan["unallocated_cash"] == pytest.approx(1000.0)

    def test_fails_when_below_min_cash_threshold(self):
        snapshot = AccountSnapshot(cash=10.0, holdings=[])
        with pytest.raises(RebalanceError):
            build_rebalance_plan(
                snapshot=snapshot,
                target_symbols=["AAPL", "MSFT", "NVDA", "AMZN", "META"],
                min_cash_threshold=100.0,
                max_order_notional=1000.0,
            )

    def test_market_value_takes_precedence_over_market_price(self):
        snapshot = AccountSnapshot(
            cash=0.0,
            holdings=[Holding(symbol="TSLA", quantity=1, market_price=10.0, market_value=50.0)],
        )
        plan = build_rebalance_plan(
            snapshot=snapshot,
            target_symbols=["AAPL", "MSFT", "NVDA", "AMZN", "META"],
            min_cash_threshold=0.0,
            max_order_notional=1000.0,
        )
        assert plan["estimated_liquidation_value"] == pytest.approx(50.0)

    def test_fails_when_holding_missing_price_and_value(self):
        snapshot = AccountSnapshot(cash=100.0, holdings=[Holding(symbol="TSLA", quantity=1)])
        with pytest.raises(RebalanceError):
            build_rebalance_plan(
                snapshot=snapshot,
                target_symbols=["AAPL", "MSFT", "NVDA", "AMZN", "META"],
                min_cash_threshold=0.0,
                max_order_notional=1000.0,
            )
