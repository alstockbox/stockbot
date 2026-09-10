from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights for the research-factory objective.

    The objective deliberately rewards return only when it survives OOS validation,
    robustness checks and risk penalties. This prevents the search process from
    selecting fragile, high-turnover curves that merely look profitable in sample.
    """

    cagr: float = 4.0
    sharpe: float = 1.5
    sortino: float = 1.0
    calmar: float = 1.0
    excess_return: float = 1.25
    robustness: float = 2.0
    oos_coverage: float = 1.0
    stress_score: float = 1.5
    max_drawdown: float = 5.0
    cvar_95: float = 6.0
    volatility: float = 0.35
    turnover: float = 0.05
    instability: float = 1.25
    concentration: float = 1.25


def _finite(metrics: dict[str, float], key: str) -> float:
    try:
        value = float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def risk_adjusted_objective(
    metrics: dict[str, float],
    *,
    robustness: float,
    oos_coverage: float,
    stress_score: float = 1.0,
    weights: ObjectiveWeights | None = None,
) -> float:
    """Score a candidate for robust, repeatable edge after costs.

    Inputs are clipped where they are naturally bounded. The function is intentionally
    monotonic in return/robustness and monotonic against drawdown/tail risk/turnover.
    It is suitable for ranking experiments, not for making live execution decisions.
    """

    w = weights or ObjectiveWeights()
    robustness = max(0.0, min(1.0, float(robustness)))
    oos_coverage = max(0.0, min(1.0, float(oos_coverage)))
    stress_score = max(0.0, min(1.0, float(stress_score)))

    reward = (
        w.cagr * _finite(metrics, "cagr")
        + w.sharpe * _finite(metrics, "sharpe")
        + w.sortino * _finite(metrics, "sortino")
        + w.calmar * _finite(metrics, "calmar")
        + w.excess_return * _finite(metrics, "excess_return")
        + w.robustness * robustness
        + w.oos_coverage * oos_coverage
        + w.stress_score * stress_score
    )
    penalty = (
        w.max_drawdown * max(0.0, _finite(metrics, "max_drawdown"))
        + w.cvar_95 * max(0.0, _finite(metrics, "cvar_95"))
        + w.volatility * max(0.0, _finite(metrics, "volatility"))
        + w.turnover * max(0.0, _finite(metrics, "turnover"))
        + w.instability * max(0.0, _finite(metrics, "instability"))
        + w.concentration * max(0.0, _finite(metrics, "concentration"))
    )
    return float(reward - penalty)
