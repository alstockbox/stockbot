from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from stockbot.evaluation.metrics import performance_metrics
from stockbot.research.regime_eval import RegimePerformanceReport, evaluate_regime_performance
from stockbot.research.stress import StressReport, evaluate_stress_suite


@dataclass(frozen=True)
class EnsembleReport:
    member_weights: dict[str, float]
    returns: pd.Series
    metrics: dict[str, float]
    stress_report: StressReport
    regime_report: RegimePerformanceReport | None
    score: float


def _softmax_weights(scores: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not scores:
        return {}
    keys = list(scores)
    values = np.asarray([float(scores[key]) for key in keys], dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("ensemble scores must be finite")
    scaled = (values - values.max()) / temperature
    exp = np.exp(np.clip(scaled, -50.0, 50.0))
    total = float(exp.sum())
    if not math.isfinite(total) or total <= 0.0:
        weights = np.full(len(keys), 1.0 / len(keys))
    else:
        weights = exp / total
    return {key: float(weight) for key, weight in zip(keys, weights)}


def _correlation_adjusted_weights(
    aligned_returns: pd.DataFrame,
    base_weights: dict[str, float],
) -> dict[str, float]:
    """Downweight redundant members while preserving score-based preference."""

    if len(base_weights) <= 1:
        return dict(base_weights)

    correlation = aligned_returns.corr(min_periods=20).abs()
    adjusted: dict[str, float] = {}
    for member, base_weight in base_weights.items():
        if member not in correlation.index:
            diversification_factor = 1.0
        else:
            peer_corr = correlation.loc[member].drop(labels=[member], errors="ignore").dropna()
            average_corr = float(peer_corr.mean()) if len(peer_corr) else 0.5
            # Low-correlation members receive more marginal weight; the 0.25 floor
            # prevents an unstable near-zero-correlation estimate from dominating.
            diversification_factor = 1.0 / (0.25 + max(0.0, min(1.0, average_corr)))
        adjusted[member] = float(base_weight) * diversification_factor

    total = sum(adjusted.values())
    if not math.isfinite(total) or total <= 0.0:
        return {member: 1.0 / len(adjusted) for member in adjusted}
    return {member: value / total for member, value in adjusted.items()}


def build_horizon_ensemble(
    returns_by_id: dict[str, pd.Series],
    promotion_scores: dict[str, float],
    *,
    regime_series: pd.Series | None = None,
    temperature: float = 0.75,
) -> EnsembleReport:
    """Blend horizon champions using score and incremental diversification value."""

    if not returns_by_id:
        raise ValueError("at least one ensemble member is required")
    if set(returns_by_id) != set(promotion_scores):
        raise ValueError("returns_by_id and promotion_scores must have identical keys")

    aligned = pd.concat(
        [pd.Series(series, dtype=float).rename(member) for member, series in returns_by_id.items()],
        axis=1,
        join="outer",
    ).sort_index()
    aligned = aligned.replace([np.inf, -np.inf], np.nan)

    base_weights = _softmax_weights(promotion_scores, temperature)
    weights = _correlation_adjusted_weights(aligned, base_weights)

    availability = aligned.notna().astype(float)
    weighted = aligned.fillna(0.0).mul(pd.Series(weights), axis=1)
    active_weight = availability.mul(pd.Series(weights), axis=1).sum(axis=1)
    ensemble_returns = weighted.sum(axis=1).div(active_weight.replace(0.0, np.nan)).fillna(0.0)
    ensemble_returns.name = "horizon_ensemble_return"

    metrics = performance_metrics(ensemble_returns)
    stress = evaluate_stress_suite(ensemble_returns)
    regime_report = None
    if regime_series is not None:
        regime_report = evaluate_regime_performance(ensemble_returns, regime_series)

    sharpe = float(metrics.get("sharpe", 0.0))
    cagr = float(metrics.get("cagr", 0.0))
    max_drawdown = max(0.0, float(metrics.get("max_drawdown", 1.0)))
    regime_score = regime_report.score if regime_report is not None else 0.5
    raw_score = (
        0.20 * float(np.clip((sharpe + 1.0) / 3.0, 0.0, 1.0))
        + 0.25 * float(np.clip(0.5 + cagr, 0.0, 1.0))
        + 0.20 * float(np.clip(1.0 - max_drawdown, 0.0, 1.0))
        + 0.20 * stress.score
        + 0.15 * regime_score
    )

    return EnsembleReport(
        member_weights=weights,
        returns=ensemble_returns,
        metrics={key: float(value) for key, value in metrics.items()},
        stress_report=stress,
        regime_report=regime_report,
        score=float(np.clip(raw_score, 0.0, 1.0)),
    )
