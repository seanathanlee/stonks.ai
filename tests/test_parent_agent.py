"""Unit tests for parent-agent forecast row construction."""

from __future__ import annotations

from agents.horizons import HORIZON_RETURN_BOUNDS, HORIZON_RETURN_KEYS
from agents.parent_agent import _build_forecast_rows


def test_build_forecast_rows_clips_expected_returns_to_horizon_bounds() -> None:
    pick = {
        "rank": 1,
        "symbol": "MXL",
        "reasoning": "unbounded extrapolation",
    }
    for horizon, key in HORIZON_RETURN_KEYS.items():
        pick[key] = HORIZON_RETURN_BOUNDS[horizon][1] * 1000

    rows = _build_forecast_rows("test_agent", [pick], "2026-05-02T00:00:00+00:00")

    assert len(rows) == len(HORIZON_RETURN_KEYS)
    for row in rows:
        assert row["expectedReturn"] == HORIZON_RETURN_BOUNDS[row["horizon"]][1]


def test_build_forecast_rows_clips_losses_to_possible_stock_loss() -> None:
    pick = {
        "rank": 1,
        "symbol": "MXL",
        "reasoning": "impossible loss",
    }
    for key in HORIZON_RETURN_KEYS.values():
        pick[key] = -10_000.0

    rows = _build_forecast_rows("test_agent", [pick], "2026-05-02T00:00:00+00:00")

    assert len(rows) == len(HORIZON_RETURN_KEYS)
    for row in rows:
        assert row["expectedReturn"] == HORIZON_RETURN_BOUNDS[row["horizon"]][0]
