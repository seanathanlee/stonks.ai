"""
Stonks.ai — Central horizon configuration.

All forecast and evaluation horizon constants live here.  To add a new
forecast horizon, add a single entry to ``FORECAST_HORIZONS``; it will
propagate automatically to:

  - ``agents/stat_agents.py``   (trading-day projection window)
  - ``agents/child_agents.py``  (LLM tool-call response schema)
  - ``agents/parent_agent.py``  (pick-dict field names)
  - ``agents/evaluation_agent.py`` (evaluation lookback window, via ALL_HORIZONS)
  - ``agents/adx_client.py``    (horizon validation)
  - ``agents/chat_agent.py``    (API tool enum values)

To add a new *evaluation-only* horizon (one not actively forecasted),
add it to ``ALL_HORIZONS`` only.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# All valid horizons
# ---------------------------------------------------------------------------
# Maps horizon label → calendar-day lookback used by the evaluation agent
# and ADX API validation.
ALL_HORIZONS: dict[str, int] = {
    "1m": 30,
    "3m": 91,
    "6m": 182,
    "1y": 365,
}

# ---------------------------------------------------------------------------
# Active forecast horizons
# ---------------------------------------------------------------------------
# Maps horizon label → approximate *trading* days in the horizon.
# Only horizons listed here will be included in agent forecasts and the
# child-agent LLM schema.
FORECAST_HORIZONS: dict[str, int] = {
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "1y": 252,
}

# ---------------------------------------------------------------------------
# Pick-dict field names
# ---------------------------------------------------------------------------
# Maps each active forecast horizon to the field name used in pick dicts,
# stat-agent output, and the ADX forecast schema.
# e.g.  "1m"  →  "expected_return_1m"
HORIZON_RETURN_KEYS: dict[str, str] = {
    h: f"expected_return_{h}" for h in FORECAST_HORIZONS
}
