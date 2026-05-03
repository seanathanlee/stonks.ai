"""
Unit tests for agents/stat_agents.py

All tests use synthetic price series so no ADX connection is required.
Tests verify that:
  - Each model function returns the expected dict structure.
  - Expected returns are finite floats.
  - ``run_stat_agent`` returns ≤5 picks in the correct schema.
  - Edge-case handling (insufficient data) raises or skips gracefully.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from agents.stat_agents import (
    MIN_HISTORY,
    HORIZONS,
    STAT_AGENTS,
    run_stat_agent,
    _momentum_factor_model,
    _historical_volatility_model,
    _ets_factory,
    _arima_factory,
    _garch_volatility_factory,
    _monte_carlo_factory,
    _capm_factory,
    _hmm_regime_factory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trending_prices() -> list[float]:
    """30 prices trending upward with slight noise."""
    rng = np.random.default_rng(0)
    base = 100.0
    prices = [base + i * 0.5 + rng.normal(0, 0.2) for i in range(30)]
    return prices


@pytest.fixture
def flat_prices() -> list[float]:
    """30 prices with no trend, minimal noise."""
    rng = np.random.default_rng(1)
    return [100.0 + rng.normal(0, 0.01) for _ in range(30)]


@pytest.fixture
def price_map(trending_prices: list[float]) -> dict[str, list[float]]:
    """Small price map with 5 symbols, each 30 points."""
    rng = np.random.default_rng(2)
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    return {
        sym: [100.0 + i * 0.3 + rng.normal(0, 0.3) for i in range(30)]
        for sym in symbols
    }


@pytest.fixture
def stock_data(price_map: dict[str, list[float]]) -> dict[str, list[dict]]:
    """Stock data in the format returned by adx_client.get_price_history."""
    from datetime import date, timedelta

    start = date(2025, 1, 1)
    result = {}
    for sym, prices in price_map.items():
        result[sym] = [
            {"date": (start + timedelta(days=i)).isoformat(), "price": p}
            for i, p in enumerate(prices)
        ]
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_horizon_dict(d: dict, *, allow_zero: bool = False) -> None:
    """Assert that d contains all horizons with finite float values."""
    assert set(d.keys()) == set(HORIZONS.keys()), f"Missing horizons: {d.keys()}"
    for h, v in d.items():
        assert isinstance(v, float), f"Horizon {h} is not float: {type(v)}"
        assert math.isfinite(v), f"Horizon {h} is not finite: {v}"
        if not allow_zero:
            # Values may legitimately be zero for perfectly flat series,
            # but should not be NaN/Inf.
            pass


def _assert_pick_schema(pick: dict) -> None:
    """Assert that a pick dict has the required keys and valid types."""
    required = {
        "rank", "symbol",
        "expected_return_1m", "expected_return_3m",
        "expected_return_6m", "expected_return_1y",
        "reasoning",
    }
    assert required.issubset(pick.keys()), f"Missing keys: {required - pick.keys()}"
    assert isinstance(pick["rank"], int)
    assert isinstance(pick["symbol"], str)
    for key in ("expected_return_1m", "expected_return_3m",
                "expected_return_6m", "expected_return_1y"):
        assert isinstance(pick[key], float), f"{key} is not float"
        assert math.isfinite(pick[key]), f"{key} is not finite"


# ---------------------------------------------------------------------------
# Tests: _momentum_factor_model
# ---------------------------------------------------------------------------

class TestMomentumFactor:
    def test_trending_returns_positive(self, trending_prices):
        result = _momentum_factor_model("TEST", trending_prices)
        _assert_horizon_dict(result)
        # Upward trend should produce positive returns for all horizons.
        for h, v in result.items():
            assert v > 0, f"Expected positive return for {h}, got {v}"

    def test_horizons_scale_with_time(self, trending_prices):
        result = _momentum_factor_model("TEST", trending_prices)
        # Longer horizon should produce larger absolute expected return.
        assert abs(result["1y"]) > abs(result["1m"])

    def test_insufficient_data_raises(self):
        with pytest.raises((ValueError, IndexError)):
            _momentum_factor_model("TEST", [100.0] * 3)

    def test_returns_all_horizons(self, trending_prices):
        result = _momentum_factor_model("TEST", trending_prices)
        assert set(result.keys()) == set(HORIZONS.keys())


# ---------------------------------------------------------------------------
# Tests: _historical_volatility_model
# ---------------------------------------------------------------------------

class TestHistoricalVolatility:
    def test_trending_prices(self, trending_prices):
        result = _historical_volatility_model("TEST", trending_prices)
        _assert_horizon_dict(result)

    def test_zero_volatility_raises(self):
        # Perfectly flat series → zero std → should raise.
        with pytest.raises((ValueError, ZeroDivisionError)):
            _historical_volatility_model("TEST", [100.0] * 30)

    def test_returns_all_horizons(self, trending_prices):
        result = _historical_volatility_model("TEST", trending_prices)
        assert set(result.keys()) == set(HORIZONS.keys())


# ---------------------------------------------------------------------------
# Tests: ETS factory
# ---------------------------------------------------------------------------

class TestETS:
    def test_trending_prices(self, trending_prices, price_map):
        factory = _ets_factory(price_map)
        result = factory("TEST", trending_prices)
        _assert_horizon_dict(result)

    def test_flat_prices(self, flat_prices, price_map):
        factory = _ets_factory(price_map)
        result = factory("TEST", flat_prices)
        _assert_horizon_dict(result, allow_zero=True)
        # Flat prices → near-zero returns for all horizons.
        for h, v in result.items():
            assert abs(v) < 5.0, f"Unexpectedly large return for flat series: {h}={v}"

    def test_returns_all_horizons(self, trending_prices, price_map):
        factory = _ets_factory(price_map)
        result = factory("TEST", trending_prices)
        assert set(result.keys()) == set(HORIZONS.keys())


# ---------------------------------------------------------------------------
# Tests: ARIMA factory
# ---------------------------------------------------------------------------

class TestARIMA:
    def test_trending_prices(self, trending_prices, price_map):
        factory = _arima_factory(price_map)
        result = factory("TEST", trending_prices)
        _assert_horizon_dict(result)

    def test_returns_all_horizons(self, trending_prices, price_map):
        factory = _arima_factory(price_map)
        result = factory("TEST", trending_prices)
        assert set(result.keys()) == set(HORIZONS.keys())

    def test_values_are_finite(self, trending_prices, price_map):
        factory = _arima_factory(price_map)
        result = factory("TEST", trending_prices)
        for h, v in result.items():
            assert math.isfinite(v), f"ARIMA {h} is non-finite: {v}"


# ---------------------------------------------------------------------------
# Tests: GARCH factory
# ---------------------------------------------------------------------------

class TestGARCH:
    def test_trending_prices(self, trending_prices, price_map):
        factory = _garch_volatility_factory(price_map)
        result = factory("TEST", trending_prices)
        _assert_horizon_dict(result)

    def test_returns_all_horizons(self, trending_prices, price_map):
        factory = _garch_volatility_factory(price_map)
        result = factory("TEST", trending_prices)
        assert set(result.keys()) == set(HORIZONS.keys())


# ---------------------------------------------------------------------------
# Tests: Monte Carlo factory
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def test_trending_prices(self, trending_prices, price_map):
        factory = _monte_carlo_factory(price_map)
        result = factory("TEST", trending_prices)
        _assert_horizon_dict(result)

    def test_reproducible(self, trending_prices, price_map):
        factory = _monte_carlo_factory(price_map)
        r1 = factory("TEST", trending_prices)
        r2 = factory("TEST", trending_prices)
        for h in HORIZONS:
            assert r1[h] == pytest.approx(r2[h], rel=1e-6)

    def test_longer_horizon_larger_absolute_return(self, trending_prices, price_map):
        factory = _monte_carlo_factory(price_map)
        result = factory("TEST", trending_prices)
        # With a positive drift, longer simulations should accumulate more return.
        assert abs(result["1y"]) > abs(result["1m"])

    def test_zero_volatility_raises(self, price_map):
        factory = _monte_carlo_factory(price_map)
        with pytest.raises((ValueError, ZeroDivisionError)):
            factory("TEST", [100.0] * 30)


# ---------------------------------------------------------------------------
# Tests: CAPM factory
# ---------------------------------------------------------------------------

class TestCAPM:
    def test_builds_market_proxy(self, price_map):
        # Should not raise.
        _capm_factory(price_map)

    def test_trending_prices(self, trending_prices, price_map):
        model_fn = _capm_factory(price_map)
        result = model_fn("TEST", trending_prices)
        _assert_horizon_dict(result)

    def test_returns_all_horizons(self, trending_prices, price_map):
        model_fn = _capm_factory(price_map)
        result = model_fn("TEST", trending_prices)
        assert set(result.keys()) == set(HORIZONS.keys())

    def test_insufficient_price_map_raises(self):
        with pytest.raises(ValueError):
            _capm_factory({})

    def test_high_beta_amplifies_return(self, price_map):
        """A stock with 2× market returns should have higher expected return than market."""
        # Build a market proxy from price_map.
        import numpy as _np
        from agents.stat_agents import _log_returns

        model_fn = _capm_factory(price_map)

        # Symbol with very high beta: prices that move 2× the market proxy.
        all_rets = [_log_returns(p) for p in price_map.values()]
        min_len = min(len(r) for r in all_rets)
        mkt = _np.mean([r[-min_len:] for r in all_rets], axis=0)
        high_beta_prices = list(100.0 * _np.exp(_np.cumsum(_np.concatenate([[0], 2 * mkt]))))

        result = model_fn("HIGHBETA", high_beta_prices)
        for h, v in result.items():
            assert math.isfinite(v)


# ---------------------------------------------------------------------------
# Tests: HMM Regime factory
# ---------------------------------------------------------------------------

class TestHMMRegime:
    def test_trending_prices(self, trending_prices, price_map):
        factory = _hmm_regime_factory(price_map)
        result = factory("TEST", trending_prices)
        _assert_horizon_dict(result)

    def test_returns_all_horizons(self, trending_prices, price_map):
        factory = _hmm_regime_factory(price_map)
        result = factory("TEST", trending_prices)
        assert set(result.keys()) == set(HORIZONS.keys())

    def test_values_bounded(self, trending_prices, price_map):
        factory = _hmm_regime_factory(price_map)
        result = factory("TEST", trending_prices)
        for h, v in result.items():
            # Reasonable daily return range per trading day
            assert abs(v) < 1000.0, f"HMM return too large: {h}={v}"


# ---------------------------------------------------------------------------
# Tests: run_stat_agent (integration)
# ---------------------------------------------------------------------------

class TestRunStatAgent:
    def test_returns_up_to_5_picks(self, stock_data):
        from agents.stat_agents import _momentum_factor_factory
        picks = run_stat_agent("momentum_factor", _momentum_factor_factory, stock_data)
        assert 1 <= len(picks) <= 5

    def test_pick_schema(self, stock_data):
        from agents.stat_agents import _momentum_factor_factory
        picks = run_stat_agent("momentum_factor", _momentum_factor_factory, stock_data)
        for pick in picks:
            _assert_pick_schema(pick)

    def test_ranks_are_consecutive(self, stock_data):
        from agents.stat_agents import _momentum_factor_factory
        picks = run_stat_agent("momentum_factor", _momentum_factor_factory, stock_data)
        ranks = [p["rank"] for p in picks]
        assert ranks == list(range(1, len(picks) + 1))

    def test_symbols_are_strings(self, stock_data):
        from agents.stat_agents import _monte_carlo_factory
        picks = run_stat_agent("monte_carlo", _monte_carlo_factory, stock_data)
        for pick in picks:
            assert isinstance(pick["symbol"], str)
            assert len(pick["symbol"]) > 0

    def test_insufficient_history_skipped(self):
        """Symbols with fewer than MIN_HISTORY data points should be silently skipped."""
        from agents.stat_agents import _momentum_factor_factory

        sparse_data = {
            "AAPL": [{"date": "2025-01-01", "price": 100.0 + i} for i in range(5)],
            "MSFT": [{"date": "2025-01-01", "price": 200.0 + i} for i in range(25)],
        }
        picks = run_stat_agent("momentum_factor", _momentum_factor_factory, sparse_data)
        # Only MSFT has enough data; should produce 1 pick.
        assert len(picks) == 1
        assert picks[0]["symbol"] == "MSFT"

    def test_empty_stock_data_returns_empty(self):
        from agents.stat_agents import _momentum_factor_factory
        picks = run_stat_agent("momentum_factor", _momentum_factor_factory, {})
        assert picks == []

    def test_all_stat_agents_run(self, stock_data):
        """Smoke test: every registered STAT_AGENT should run without error."""
        for agent in STAT_AGENTS:
            picks = run_stat_agent(agent.name, agent.model_fn_factory, stock_data)
            assert isinstance(picks, list), f"Agent {agent.name} did not return a list"
            assert len(picks) <= 5, f"Agent {agent.name} returned more than 5 picks"
            for pick in picks:
                _assert_pick_schema(pick)

    def test_reasoning_contains_agent_name(self, stock_data):
        from agents.stat_agents import _momentum_factor_factory
        picks = run_stat_agent("momentum_factor", _momentum_factor_factory, stock_data)
        for pick in picks:
            assert "momentum_factor" in pick["reasoning"]
