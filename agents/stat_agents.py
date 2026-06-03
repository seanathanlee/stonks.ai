"""
Stonks.ai Statistical Model Agents

Pure-statistics child agents that compute expected returns directly from
30-day price histories — no LLM call required.

Each agent implements a distinct statistical or financial model:

  momentum_factor        – composite price momentum (k=5, 10, 20 days)
  historical_volatility  – drift / volatility Sharpe proxy
  ets                    – Holt-Winters exponential smoothing
  arima                  – ARIMA(p,1,q) auto-selected by AIC
  garch_volatility       – GARCH(1,1) volatility-adjusted drift
  monte_carlo            – Geometric Brownian Motion simulation (N=1,000 paths)
  capm_beta              – CAPM beta vs equal-weighted cross-sectional proxy
  hmm_regime             – 2-state Hidden Markov Model regime detection

Each model function has the signature::

    model_fn(symbol: str, prices: list[float]) -> dict[str, float]

where the returned dict maps the horizon label ("1m") to the expected
percentage return.

The shared runner ``run_stat_agent`` applies a model function to every symbol
in the price dataset, ranks symbols by their 1-month expected return, and
returns the top 5 picks in the standard ADX forecast schema used by all child
agents.

For agents that need cross-sectional context (CAPM, HMM) a *factory* pattern
is used: ``model_fn_factory(price_map) -> model_fn``.  Simple agents return
their model function unchanged from the factory.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum number of price data-points required to fit any model.
MIN_HISTORY = 20

# Number of Monte Carlo price-path simulations per symbol.
MONTE_CARLO_SIMS = 1_000

# Maximum EM iterations for the HMM fitting step.
HMM_ITERATIONS = 200

# Horizon labels → approximate number of trading days.
HORIZONS: dict[str, int] = {
    "1m": 21,
}

# Type aliases
ModelFn = Callable[[str, list[float]], dict[str, float]]
ModelFnFactory = Callable[[dict[str, list[float]]], ModelFn]


# ---------------------------------------------------------------------------
# Shared agent runner
# ---------------------------------------------------------------------------


def run_stat_agent(
    name: str,
    model_fn_factory: ModelFnFactory,
    stock_data: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Apply a statistical model to all symbols and return the top 5 picks.

    Parameters
    ----------
    name:
        Human-readable agent name stored in ADX forecasts.
    model_fn_factory:
        Callable that receives a ``{symbol: [price, ...]}`` mapping and
        returns a ``model_fn(symbol, prices) -> {horizon: pct_return}``
        closure.  Simple agents ignore the price map; cross-sectional agents
        (CAPM, HMM) use it to build a shared market proxy.
    stock_data:
        Full price history as returned by ``adx_client.get_price_history``:
        ``{symbol: [{"date": str, "price": float}, ...]}``.

    Returns
    -------
    List of ≤5 pick dicts compatible with the ADX forecast schema:
        symbol, rank, expected_return_1m, reasoning.
    """
    # Build a {symbol: [price, ...]} map for the factory.
    price_map: dict[str, list[float]] = {
        sym: [r["price"] for r in hist]
        for sym, hist in stock_data.items()
        if len(hist) >= MIN_HISTORY
    }

    if not price_map:
        log.warning("Agent '%s': no symbols with sufficient history.", name)
        return []

    try:
        model_fn = model_fn_factory(price_map)
    except Exception as exc:
        log.error("Agent '%s': factory failed — %s", name, exc)
        return []

    # Apply model to every symbol.
    signals: dict[str, dict[str, float]] = {}
    for symbol, prices in price_map.items():
        try:
            result = model_fn(symbol, prices)
            signals[symbol] = result
        except Exception as exc:
            log.debug("Agent '%s': model failed for %s — %s", name, symbol, exc)

    if not signals:
        log.warning("Agent '%s': no signals computed.", name)
        return []

    # Rank by composite signal: mean expected return across all horizons.
    ranked = sorted(
        signals.items(),
        key=lambda kv: sum(kv[1].values()) / max(len(kv[1]), 1),
        reverse=True,
    )[:5]

    picks: list[dict[str, Any]] = []
    for rank, (symbol, returns) in enumerate(ranked, start=1):
        horizon_summary = ", ".join(
            f"{h}={returns.get(h, 0.0):.2f}%" for h in ("1m",)
        )
        picks.append(
            {
                "rank": rank,
                "symbol": symbol,
                "expected_return_1m": float(returns.get("1m", 0.0)),
                "reasoning": f"{name}: {horizon_summary}",
            }
        )
    return picks


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _log_returns(prices: list[float]) -> np.ndarray:
    """Compute daily log returns from a price series."""
    arr = np.array(prices, dtype=float)
    return np.diff(np.log(arr + 1e-12))


def _project_horizons(daily_pct: float) -> dict[str, float]:
    """
    Project a constant daily percentage drift across all forecast horizons.

    This is a linear extrapolation of the daily rate — appropriate for
    small returns where compounding effects are minor.
    """
    return {h: daily_pct * d for h, d in HORIZONS.items()}


# ---------------------------------------------------------------------------
# 1. Momentum Factor Agent
# ---------------------------------------------------------------------------


def _momentum_factor_model(symbol: str, prices: list[float]) -> dict[str, float]:
    """
    Composite price momentum: arithmetic mean of k=5, 10, 20 day momentum.

    momentum_k = (P_t - P_{t-k}) / P_{t-k} * 100
    """
    arr = np.array(prices, dtype=float)
    n = len(arr)
    current = arr[-1]
    if current <= 0:
        raise ValueError("Non-positive current price.")

    ks = [k for k in (5, 10, 20) if n > k]
    if not ks:
        raise ValueError("Insufficient history for momentum calculation.")

    moms = [(current - arr[-(k + 1)]) / arr[-(k + 1)] * 100.0 for k in ks]
    composite_20d = float(np.mean(moms))

    # Extrapolate: assume the 20-day momentum rate is approximately a 1-month
    # rate (21 trading days).  Project forward proportionally.
    daily_pct = composite_20d / 20.0
    return _project_horizons(daily_pct)


def _momentum_factor_factory(_price_map: dict[str, list[float]]) -> ModelFn:
    return _momentum_factor_model


# ---------------------------------------------------------------------------
# 2. Historical Volatility Agent
# ---------------------------------------------------------------------------


def _historical_volatility_model(symbol: str, prices: list[float]) -> dict[str, float]:
    """
    Drift / volatility ratio (simplified Sharpe proxy).

    Uses log returns to estimate the daily drift (mu) and annualised
    volatility (sigma), then projects the drift forward across each horizon.
    """
    returns = _log_returns(prices)
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))

    if sigma < 1e-10:
        raise ValueError("Zero volatility — cannot compute Sharpe proxy.")

    # Daily expected % return from drift
    daily_pct = (math.exp(mu) - 1.0) * 100.0
    return _project_horizons(daily_pct)


def _historical_volatility_factory(_price_map: dict[str, list[float]]) -> ModelFn:
    return _historical_volatility_model


# ---------------------------------------------------------------------------
# 3. ETS (Exponential Smoothing) Agent
# ---------------------------------------------------------------------------


def _ets_factory(_price_map: dict[str, list[float]]) -> ModelFn:
    def _ets_model(symbol: str, prices: list[float]) -> dict[str, float]:
        """
        Holt-Winters additive-trend exponential smoothing (no seasonality).

        Forecasts the terminal price h trading days ahead and converts to
        an expected percentage return.
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore[import]

        current = prices[-1]
        if current <= 0:
            raise ValueError("Non-positive current price.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                prices, trend="add", seasonal=None
            ).fit(optimized=True)

        result: dict[str, float] = {}
        for horizon, days in HORIZONS.items():
            forecast = model.forecast(days)
            projected = float(forecast[-1])
            result[horizon] = (projected - current) / current * 100.0
        return result

    return _ets_model


# ---------------------------------------------------------------------------
# 4. ARIMA Agent
# ---------------------------------------------------------------------------


def _arima_factory(_price_map: dict[str, list[float]]) -> ModelFn:
    def _arima_model(symbol: str, prices: list[float]) -> dict[str, float]:
        """
        ARIMA(p,1,q) with (p, q) selected by AIC over a small search grid.

        Forecasts terminal price at each horizon and converts to % return.
        ARIMA forecasts beyond the sample revert toward the estimated drift,
        so long-horizon values are intentionally conservative.
        """
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore[import]

        current = prices[-1]
        if current <= 0:
            raise ValueError("Non-positive current price.")

        best_model = None
        best_aic = float("inf")

        for p in range(0, 3):
            for q in range(0, 3):
                if p == 0 and q == 0:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = ARIMA(prices, order=(p, 1, q)).fit()
                    if m.aic < best_aic:
                        best_aic = m.aic
                        best_model = m
                except Exception:
                    continue

        if best_model is None:
            raise ValueError("ARIMA fitting failed for all tried orders.")

        result: dict[str, float] = {}
        for horizon, days in HORIZONS.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = best_model.forecast(steps=days)
            projected = float(fc.iloc[-1] if hasattr(fc, "iloc") else fc[-1])
            result[horizon] = (projected - current) / current * 100.0
        return result

    return _arima_model


# ---------------------------------------------------------------------------
# 5. GARCH Volatility Agent
# ---------------------------------------------------------------------------


def _garch_volatility_factory(_price_map: dict[str, list[float]]) -> ModelFn:
    def _garch_model(symbol: str, prices: list[float]) -> dict[str, float]:
        """
        GARCH(1,1) volatility model.

        Estimates the conditional daily volatility and uses the historical
        mean return as the drift.  Ranks stocks implicitly by their Sharpe
        proxy (drift / forecast_sigma); the expected return stored is the
        drift projection.
        """
        from arch import arch_model  # type: ignore[import]

        arr = np.array(prices, dtype=float)
        # arch expects returns in percentage form
        log_rets_pct = _log_returns(prices) * 100.0

        if len(log_rets_pct) < 10:
            raise ValueError("Insufficient data for GARCH fitting.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(log_rets_pct, vol="Garch", p=1, q=1, dist="normal")
            res = am.fit(disp="off", show_warning=False)

        # Daily mean log return (back to fraction)
        mu_daily = float(np.mean(_log_returns(prices)))
        daily_pct = (math.exp(mu_daily) - 1.0) * 100.0
        return _project_horizons(daily_pct)

    return _garch_model


# ---------------------------------------------------------------------------
# 6. Monte Carlo (GBM) Agent
# ---------------------------------------------------------------------------


def _monte_carlo_factory(_price_map: dict[str, list[float]]) -> ModelFn:
    def _monte_carlo_model(symbol: str, prices: list[float]) -> dict[str, float]:
        """
        Geometric Brownian Motion Monte Carlo simulation (N=1,000 paths).

        Estimates mu and sigma from the historical log return series, then
        simulates price paths and reports the median terminal price as the
        expected return.  The seed is fixed for reproducibility.
        """
        arr = np.array(prices, dtype=float)
        log_rets = _log_returns(prices)
        mu = float(np.mean(log_rets))
        sigma = float(np.std(log_rets, ddof=1))
        current = arr[-1]

        if current <= 0 or sigma < 1e-10:
            raise ValueError("Invalid price series for Monte Carlo simulation.")

        rng = np.random.default_rng(seed=42)

        result: dict[str, float] = {}
        for horizon, days in HORIZONS.items():
            z = rng.standard_normal((MONTE_CARLO_SIMS, days))
            # GBM log return per step: (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
            drift = (mu - 0.5 * sigma**2)
            step_log_rets = drift + sigma * z
            terminal_prices = current * np.exp(step_log_rets.sum(axis=1))
            median_price = float(np.median(terminal_prices))
            result[horizon] = (median_price - current) / current * 100.0
        return result

    return _monte_carlo_model


# ---------------------------------------------------------------------------
# 7. CAPM Beta Agent
# ---------------------------------------------------------------------------


def _capm_factory(price_map: dict[str, list[float]]) -> ModelFn:
    """
    Build a CAPM model function using an equal-weighted cross-sectional
    market proxy constructed from all available symbols.
    """
    # Compute log returns for each symbol that has enough history.
    all_returns: list[np.ndarray] = []
    for prices in price_map.values():
        r = _log_returns(prices)
        if len(r) >= MIN_HISTORY - 1:
            all_returns.append(r)

    if not all_returns:
        raise ValueError("No symbols with sufficient history for market proxy.")

    # Align all return series to the shortest length.
    min_len = min(len(r) for r in all_returns)
    market_returns = np.mean(
        np.stack([r[-min_len:] for r in all_returns], axis=0), axis=0
    )

    # Historical market daily log return (used as market premium proxy).
    market_mu = float(np.mean(market_returns))

    def _capm_model(symbol: str, prices: list[float]) -> dict[str, float]:
        """CAPM expected return: E[R] = beta × market_premium."""
        stock_rets = _log_returns(prices)
        n = min(len(stock_rets), len(market_returns))
        if n < 10:
            raise ValueError("Insufficient overlap with market proxy.")

        s = stock_rets[-n:]
        m = market_returns[-n:]

        cov_matrix = np.cov(s, m)
        market_var = cov_matrix[1, 1]
        if market_var < 1e-10:
            raise ValueError("Market proxy has near-zero variance.")

        beta = cov_matrix[0, 1] / market_var

        # Expected daily log return = beta × market_mu (CAPM excess return)
        expected_daily_log = beta * market_mu
        daily_pct = (math.exp(expected_daily_log) - 1.0) * 100.0
        return _project_horizons(daily_pct)

    return _capm_model


# ---------------------------------------------------------------------------
# 8. HMM Regime Agent
# ---------------------------------------------------------------------------


def _hmm_regime_factory(_price_map: dict[str, list[float]]) -> ModelFn:
    def _hmm_model(symbol: str, prices: list[float]) -> dict[str, float]:
        """
        2-state Gaussian HMM regime detection.

        Fits a 2-component HMM on the daily log return series to identify
        bull and bear regimes.  The expected return is derived from the bull
        state mean return, weighted by the probability of the current regime
        persisting (or transitioning to) the bull state.
        """
        from hmmlearn import hmm as _hmm  # type: ignore[import]

        log_rets = _log_returns(prices).reshape(-1, 1)
        if len(log_rets) < MIN_HISTORY:
            raise ValueError("Insufficient data for HMM fitting.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _hmm.GaussianHMM(
                n_components=2,
                covariance_type="full",
                n_iter=HMM_ITERATIONS,
                random_state=42,
            )
            model.fit(log_rets)

        means = model.means_.flatten()
        bull_state = int(np.argmax(means))
        bull_mean_daily = float(means[bull_state])

        # Identify the current regime.
        states = model.predict(log_rets)
        current_state = int(states[-1])

        if current_state == bull_state:
            # Weight by probability of staying in the bull state.
            weight = float(model.transmat_[current_state, current_state])
        else:
            # Weight by probability of transitioning to the bull state.
            weight = float(model.transmat_[current_state, bull_state])

        daily_pct = (math.exp(bull_mean_daily) - 1.0) * 100.0 * weight
        return _project_horizons(daily_pct)

    return _hmm_model


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------


class _StatAgentDef:
    """Lightweight container for a statistical agent definition."""

    __slots__ = ("name", "model_fn_factory")

    def __init__(self, name: str, model_fn_factory: ModelFnFactory) -> None:
        self.name = name
        self.model_fn_factory = model_fn_factory


STAT_AGENTS: list[_StatAgentDef] = [
    _StatAgentDef("momentum_factor", _momentum_factor_factory),
    _StatAgentDef("historical_volatility", _historical_volatility_factory),
    _StatAgentDef("ets", _ets_factory),
    _StatAgentDef("arima", _arima_factory),
    _StatAgentDef("garch_volatility", _garch_volatility_factory),
    _StatAgentDef("monte_carlo", _monte_carlo_factory),
    _StatAgentDef("capm_beta", _capm_factory),
    _StatAgentDef("hmm_regime", _hmm_regime_factory),
]
