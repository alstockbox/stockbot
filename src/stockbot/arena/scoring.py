from __future__ import annotations

import math


def _m(metrics: dict[str, float], key: str) -> float:
    value = float(metrics.get(key, 0.0))
    return value if math.isfinite(value) else 0.0


def research_score(metrics: dict[str, float], robustness: float) -> float:
    robustness = max(0.0, min(1.0, float(robustness)))
    reward = (
        3.0 * _m(metrics, "cagr")
        + 1.2 * _m(metrics, "sharpe")
        + 0.8 * _m(metrics, "sortino")
        + 0.7 * _m(metrics, "calmar")
        + 1.0 * _m(metrics, "excess_return")
        + 1.5 * robustness
    )
    penalty = (
        4.0 * _m(metrics, "max_drawdown")
        + 5.0 * _m(metrics, "cvar_95")
        + 0.08 * _m(metrics, "turnover")
        + 1.0 * _m(metrics, "instability")
        + 1.0 * _m(metrics, "concentration")
    )
    return float(reward - penalty)
