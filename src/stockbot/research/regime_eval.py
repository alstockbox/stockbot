from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockbot.domain.models import MarketRegime
from stockbot.evaluation.metrics import performance_metrics


@dataclass(frozen=True)
class RegimePerformanceReport:
    metrics_by_regime: dict[str, dict[str, float]]
    observations_by_regime: dict[str, int]
    score_by_regime: dict[str, float]
    score: float
    coverage: float
    worst_sharpe: float


def build_market_regime_series(bars: pd.DataFrame) -> pd.Series:
    """Build a causal equal-weight market regime label from the traded universe."""

    required = {"symbol", "timestamp", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")

    frame = bars.loc[:, ["symbol", "timestamp", "close"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    matrix = frame.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    if matrix.empty:
        return pd.Series(dtype="object", name="market_regime")

    asset_returns = matrix.pct_change()
    market_return = asset_returns.mean(axis=1, skipna=True).fillna(0.0)
    momentum_20 = (1.0 + market_return).rolling(20, min_periods=20).apply(np.prod, raw=True) - 1.0
    vol_20 = market_return.rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)

    regime = pd.Series(MarketRegime.NEUTRAL_CHOP.value, index=matrix.index, dtype="object")
    bear = (vol_20 >= 0.45) | (momentum_20 <= -0.04)
    bull = (momentum_20 >= 0.03) & (vol_20 < 0.35)
    regime.loc[bear.fillna(False)] = MarketRegime.BEAR_STRESS.value
    regime.loc[bull.fillna(False) & ~bear.fillna(False)] = MarketRegime.BULL_TREND.value
    regime.name = "market_regime"
    return regime


def _bounded_regime_score(metrics: dict[str, float]) -> float:
    sharpe = float(metrics.get("sharpe", 0.0))
    cagr = float(metrics.get("cagr", 0.0))
    max_drawdown = max(0.0, float(metrics.get("max_drawdown", 1.0)))
    raw = 0.50 + 0.12 * sharpe + 0.35 * cagr - 0.50 * max_drawdown
    return float(np.clip(raw, 0.0, 1.0))


def evaluate_regime_performance(
    returns: pd.Series,
    regime_series: pd.Series,
    *,
    min_observations: int = 10,
) -> RegimePerformanceReport:
    """Measure whether a candidate's OOS edge survives different market regimes."""

    if min_observations <= 0:
        raise ValueError("min_observations must be positive")

    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan)
    regimes = pd.Series(regime_series).reindex(r.index)
    aligned = pd.DataFrame({"return": r, "regime": regimes}).dropna()

    metrics_by_regime: dict[str, dict[str, float]] = {}
    observations: dict[str, int] = {}
    scores: dict[str, float] = {}
    sharpes: list[float] = []
    weighted_score = 0.0
    weighted_n = 0

    all_regimes = [regime.value for regime in MarketRegime]
    for regime_name in all_regimes:
        subset = aligned.loc[aligned["regime"] == regime_name, "return"]
        count = int(len(subset))
        observations[regime_name] = count
        if count < min_observations:
            continue
        metrics = performance_metrics(subset)
        metrics_by_regime[regime_name] = metrics
        score = _bounded_regime_score(metrics)
        scores[regime_name] = score
        sharpes.append(float(metrics.get("sharpe", 0.0)))
        weighted_score += score * count
        weighted_n += count

    available = len(scores)
    coverage = available / len(all_regimes)
    if not scores:
        aggregate_score = 0.0
        worst_sharpe = 0.0
    else:
        mean_score = weighted_score / max(1, weighted_n)
        worst_score = min(scores.values())
        aggregate_score = float(np.clip(0.60 * mean_score + 0.40 * worst_score, 0.0, 1.0))
        worst_sharpe = min(sharpes)

    return RegimePerformanceReport(
        metrics_by_regime=metrics_by_regime,
        observations_by_regime=observations,
        score_by_regime=scores,
        score=aggregate_score,
        coverage=float(coverage),
        worst_sharpe=float(worst_sharpe),
    )
