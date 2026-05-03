"""
Unit tests for agents/evaluation_agent.py

All tests use synthetic data so no ADX connection is required.
"""

from __future__ import annotations

import math
import pytest

from agents.evaluation_agent import (
    HORIZON_LOOKBACK_DAYS,
    DEFAULT_HORIZON,
    _forecast_date,
    _compute_actual_return,
    _compute_realized_volatility,
    _spearman_rho,
    _assign_actual_ranks,
    _build_evaluation_rows,
    _enrich_rows_with_run_metrics,
)
from datetime import date, timedelta


# ---------------------------------------------------------------------------
# _forecast_date
# ---------------------------------------------------------------------------

class TestForecastDate:
    def test_default_horizon_is_30_days(self):
        ref = date(2026, 5, 3)
        result = _forecast_date("1m", today=ref)
        assert result == "2026-04-03"

    def test_3m_horizon(self):
        ref = date(2026, 5, 3)
        result = _forecast_date("3m", today=ref)
        expected = (ref - timedelta(days=91)).isoformat()
        assert result == expected

    def test_1y_horizon(self):
        ref = date(2026, 5, 3)
        result = _forecast_date("1y", today=ref)
        expected = (ref - timedelta(days=365)).isoformat()
        assert result == expected

    def test_all_horizons_defined(self):
        for h in ("1m", "3m", "6m", "1y"):
            assert h in HORIZON_LOOKBACK_DAYS


# ---------------------------------------------------------------------------
# _compute_actual_return
# ---------------------------------------------------------------------------

class TestComputeActualReturn:
    def test_positive_return(self):
        assert _compute_actual_return(100.0, 110.0) == pytest.approx(10.0)

    def test_negative_return(self):
        assert _compute_actual_return(100.0, 90.0) == pytest.approx(-10.0)

    def test_zero_start_price(self):
        assert _compute_actual_return(0.0, 50.0) == 0.0

    def test_no_change(self):
        assert _compute_actual_return(100.0, 100.0) == 0.0


# ---------------------------------------------------------------------------
# _compute_realized_volatility
# ---------------------------------------------------------------------------

def _make_prices(values: list[float]) -> list[dict]:
    return [{"date": f"2026-01-{i+1:02d}", "price": v} for i, v in enumerate(values)]


class TestComputeRealizedVolatility:
    def test_too_few_points_returns_none(self):
        assert _compute_realized_volatility(_make_prices([100.0, 101.0])) is None

    def test_constant_prices_returns_zero_or_near_zero(self):
        prices = _make_prices([100.0] * 10)
        vol = _compute_realized_volatility(prices)
        # log returns are all 0 → variance is 0 → vol is 0
        assert vol == 0.0

    def test_volatile_series_returns_positive(self):
        import random
        random.seed(42)
        prices = [100.0]
        for _ in range(29):
            prices.append(prices[-1] * (1 + random.uniform(-0.05, 0.05)))
        vol = _compute_realized_volatility(_make_prices(prices))
        assert vol is not None
        assert vol > 0.0

    def test_result_is_annualised_percentage(self):
        # A daily return of exactly 1% every day → daily log return ≈ 0.00995
        # annualised vol ≈ 0.00995 * sqrt(252) * 100 ≈ 15.8%  (but std=0 for constant series)
        # Use alternating +1%/-1% to get non-zero std.
        prices_vals = [100.0]
        for i in range(19):
            factor = 1.01 if i % 2 == 0 else 1 / 1.01
            prices_vals.append(prices_vals[-1] * factor)
        vol = _compute_realized_volatility(_make_prices(prices_vals))
        assert vol is not None
        assert vol > 0.0


# ---------------------------------------------------------------------------
# _spearman_rho
# ---------------------------------------------------------------------------

class TestSpearmanRho:
    def test_perfect_positive_correlation(self):
        rho = _spearman_rho([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert rho == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        rho = _spearman_rho([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        assert rho == pytest.approx(-1.0)

    def test_no_correlation(self):
        # Ranks 1-4 vs 2,1,4,3 — moderate correlation, but not perfect
        rho = _spearman_rho([1, 2, 3, 4], [2, 1, 4, 3])
        assert rho is not None
        assert -1.0 <= rho <= 1.0

    def test_single_element_returns_none(self):
        assert _spearman_rho([1], [1]) is None

    def test_mismatched_lengths_returns_none(self):
        assert _spearman_rho([1, 2], [1, 2, 3]) is None

    def test_all_same_forecast_ranks_returns_none(self):
        # Zero variance in forecast ranks → undefined correlation
        assert _spearman_rho([3, 3, 3], [1, 2, 3]) is None


# ---------------------------------------------------------------------------
# _assign_actual_ranks
# ---------------------------------------------------------------------------

class TestAssignActualRanks:
    def test_basic_ranking(self):
        returns = {"A": 10.0, "B": 5.0, "C": 15.0}
        ranks = _assign_actual_ranks(returns)
        assert ranks["C"] == 1
        assert ranks["A"] == 2
        assert ranks["B"] == 3

    def test_ties_get_same_rank(self):
        returns = {"A": 10.0, "B": 10.0, "C": 5.0}
        ranks = _assign_actual_ranks(returns)
        assert ranks["A"] == ranks["B"] == 1
        assert ranks["C"] == 3

    def test_single_symbol(self):
        ranks = _assign_actual_ranks({"X": 7.0})
        assert ranks["X"] == 1


# ---------------------------------------------------------------------------
# _build_evaluation_rows
# ---------------------------------------------------------------------------

def _make_forecasts(agent: str, symbols_ranks: list[tuple[str, int, float]]) -> list[dict]:
    """Helper: build minimal forecast dicts."""
    return [
        {
            "agentName": agent,
            "symbol": sym,
            "rank": rank,
            "expectedReturn": exp_ret,
            "reportTime": "2026-04-03T00:00:00+00:00",
            "horizon": "1m",
        }
        for sym, rank, exp_ret in symbols_ranks
    ]


class TestBuildEvaluationRows:
    def _run(self, forecasts, actual_returns, actual_ranks, volatilities=None):
        vols = volatilities or {sym: None for sym in actual_returns}
        return _build_evaluation_rows(
            forecasts, actual_returns, actual_ranks, vols,
            report_time="2026-05-03T00:00:00+00:00",
            run_id="test-run-id",
        )

    def test_basic_row_structure(self):
        forecasts = _make_forecasts("agent_a", [("AAPL", 1, 5.0)])
        actual_returns = {"AAPL": 8.0}
        actual_ranks = {"AAPL": 1}
        rows = self._run(forecasts, actual_returns, actual_ranks)
        assert len(rows) == 1
        row = rows[0]
        required_keys = {
            "symbol", "forecastReturn", "actualReturn", "forecastRank", "actualRank",
            "accuracyScore", "returnMAE", "returnBias", "directionCorrect",
            "volatilityAdjustedError", "spearmanRho", "precisionAt5",
            "agentName", "forecastReportTime", "reportTime", "runId", "horizon",
        }
        assert required_keys.issubset(row.keys())

    def test_return_mae_equals_abs_error(self):
        forecasts = _make_forecasts("agent_a", [("AAPL", 1, 5.0)])
        rows = self._run(forecasts, {"AAPL": 8.0}, {"AAPL": 1})
        assert rows[0]["returnMAE"] == pytest.approx(3.0)

    def test_return_bias_is_signed(self):
        forecasts = _make_forecasts("agent_a", [("AAPL", 1, 5.0)])
        rows = self._run(forecasts, {"AAPL": 8.0}, {"AAPL": 1})
        # forecastReturn(5) - actualReturn(8) = -3
        assert rows[0]["returnBias"] == pytest.approx(-3.0)

    def test_direction_correct_both_positive(self):
        forecasts = _make_forecasts("a", [("X", 1, 2.0)])
        rows = self._run(forecasts, {"X": 5.0}, {"X": 1})
        assert rows[0]["directionCorrect"] == 1

    def test_direction_correct_both_negative(self):
        forecasts = _make_forecasts("a", [("X", 1, -2.0)])
        rows = self._run(forecasts, {"X": -5.0}, {"X": 1})
        assert rows[0]["directionCorrect"] == 1

    def test_direction_incorrect_opposite_signs(self):
        forecasts = _make_forecasts("a", [("X", 1, 3.0)])
        rows = self._run(forecasts, {"X": -1.0}, {"X": 3})
        assert rows[0]["directionCorrect"] == 0

    def test_volatility_adjusted_error_computed(self):
        forecasts = _make_forecasts("a", [("X", 1, 10.0)])
        vols = {"X": 20.0}  # 20% annualised vol
        rows = self._run(forecasts, {"X": 5.0}, {"X": 1}, volatilities=vols)
        # MAE = 5.0, vol = 20.0 → adjusted = 0.25
        assert rows[0]["volatilityAdjustedError"] == pytest.approx(0.25)

    def test_volatility_adjusted_error_none_when_no_vol(self):
        forecasts = _make_forecasts("a", [("X", 1, 10.0)])
        rows = self._run(forecasts, {"X": 5.0}, {"X": 1}, volatilities={"X": None})
        assert rows[0]["volatilityAdjustedError"] is None

    def test_missing_symbol_skipped(self):
        forecasts = _make_forecasts("a", [("MISSING", 1, 5.0)])
        rows = self._run(forecasts, {}, {})
        assert rows == []

    def test_spearman_and_precision_initially_none(self):
        forecasts = _make_forecasts("a", [("X", 1, 5.0)])
        rows = self._run(forecasts, {"X": 5.0}, {"X": 1})
        assert rows[0]["spearmanRho"] is None
        assert rows[0]["precisionAt5"] is None


# ---------------------------------------------------------------------------
# _enrich_rows_with_run_metrics
# ---------------------------------------------------------------------------

class TestEnrichRowsWithRunMetrics:
    def _base_row(self, agent, symbol, forecast_rank, actual_rank):
        return {
            "agentName": agent,
            "symbol": symbol,
            "forecastRank": forecast_rank,
            "actualRank": actual_rank,
            "spearmanRho": None,
            "precisionAt5": None,
        }

    def test_spearman_perfect_correlation(self):
        rows = [
            self._base_row("a", "A", 1, 1),
            self._base_row("a", "B", 2, 2),
            self._base_row("a", "C", 3, 3),
            self._base_row("a", "D", 4, 4),
            self._base_row("a", "E", 5, 5),
        ]
        actual_ranks = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
        _enrich_rows_with_run_metrics(rows, actual_ranks)
        assert all(r["spearmanRho"] == pytest.approx(1.0) for r in rows)

    def test_spearman_perfect_anti_correlation(self):
        rows = [
            self._base_row("a", "A", 1, 5),
            self._base_row("a", "B", 2, 4),
            self._base_row("a", "C", 3, 3),
            self._base_row("a", "D", 4, 2),
            self._base_row("a", "E", 5, 1),
        ]
        actual_ranks = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
        _enrich_rows_with_run_metrics(rows, actual_ranks)
        assert all(r["spearmanRho"] == pytest.approx(-1.0) for r in rows)

    def test_precision_at5_full_overlap(self):
        # Agent's top-5 exactly matches actual top-5
        rows = [self._base_row("a", sym, i + 1, i + 1) for i, sym in enumerate("ABCDE")]
        actual_ranks = {sym: i + 1 for i, sym in enumerate("ABCDE")}
        _enrich_rows_with_run_metrics(rows, actual_ranks)
        assert all(r["precisionAt5"] == pytest.approx(1.0) for r in rows)

    def test_precision_at5_zero_overlap(self):
        # Agent picks symbols that are all outside the actual top-5
        rows = [self._base_row("a", sym, i + 1, i + 6) for i, sym in enumerate("ABCDE")]
        actual_ranks = {sym: i + 6 for i, sym in enumerate("ABCDE")}
        _enrich_rows_with_run_metrics(rows, actual_ranks)
        assert all(r["precisionAt5"] == pytest.approx(0.0) for r in rows)

    def test_two_agents_enriched_independently(self):
        rows_a = [self._base_row("agent_a", "A", 1, 1), self._base_row("agent_a", "B", 2, 2)]
        rows_b = [self._base_row("agent_b", "C", 1, 5), self._base_row("agent_b", "D", 2, 6)]
        all_rows = rows_a + rows_b
        actual_ranks = {"A": 1, "B": 2, "C": 5, "D": 6}
        _enrich_rows_with_run_metrics(all_rows, actual_ranks)
        # Both agent_a rows get the same rho
        assert rows_a[0]["spearmanRho"] == rows_a[1]["spearmanRho"]
        # agent_b rows are independent
        assert rows_b[0]["spearmanRho"] == rows_b[1]["spearmanRho"]

    def test_single_row_agent_spearman_is_none(self):
        rows = [self._base_row("a", "X", 1, 3)]
        actual_ranks = {"X": 3}
        _enrich_rows_with_run_metrics(rows, actual_ranks)
        # Only 1 data point → spearman undefined
        assert rows[0]["spearmanRho"] is None

    def test_precision_partial_overlap(self):
        # 3 out of 5 picks are in actual top-5
        rows = [
            self._base_row("a", "A", 1, 1),  # in actual top-5
            self._base_row("a", "B", 2, 2),  # in actual top-5
            self._base_row("a", "C", 3, 3),  # in actual top-5
            self._base_row("a", "D", 4, 6),  # NOT in actual top-5
            self._base_row("a", "E", 5, 7),  # NOT in actual top-5
        ]
        actual_ranks = {"A": 1, "B": 2, "C": 3, "D": 6, "E": 7, "F": 4, "G": 5}
        _enrich_rows_with_run_metrics(rows, actual_ranks)
        assert all(r["precisionAt5"] == pytest.approx(3 / 5) for r in rows)
