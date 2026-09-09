from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PaperReadinessReport:
    ready: bool
    score: float
    reasons: tuple[str, ...]
    components: dict[str, float]


def _finite_or(value: float, fallback: float) -> tuple[float, bool]:
    parsed = float(value)
    if np.isfinite(parsed):
        return parsed, True
    return float(fallback), False


def evaluate_paper_readiness(
    *,
    oos_coverage: float,
    robustness: float,
    stress_score: float,
    regime_score: float,
    discovery_q_value: float | None,
    drift_score: float,
    drift_degraded: bool,
    holdout_passed: bool,
    holdout_score: float | None,
    min_oos_coverage: float = 0.40,
    min_robustness: float = 0.55,
    min_stress_score: float = 0.45,
    min_regime_score: float = 0.20,
    max_discovery_q_value: float = 0.20,
    min_drift_score: float = 0.50,
) -> PaperReadinessReport:
    """Aggregate the research evidence required before extended paper trading."""

    oos_value, oos_finite = _finite_or(oos_coverage, 0.0)
    robustness_value, robustness_finite = _finite_or(robustness, 0.0)
    stress_value, stress_finite = _finite_or(stress_score, 0.0)
    regime_value, regime_finite = _finite_or(regime_score, 0.0)
    drift_value, drift_finite = _finite_or(drift_score, 0.0)

    if discovery_q_value is None:
        q_value = 1.0
        q_finite = True
    else:
        q_value, q_finite = _finite_or(discovery_q_value, 1.0)

    if holdout_score is None:
        holdout_value = None
        holdout_finite = True
    else:
        holdout_value, holdout_finite = _finite_or(holdout_score, -1.0)

    finite_evidence = all(
        (
            oos_finite,
            robustness_finite,
            stress_finite,
            regime_finite,
            q_finite,
            drift_finite,
            holdout_finite,
        )
    )
    holdout_component = (
        0.0
        if holdout_value is None
        else float(np.clip((holdout_value + 1.0) / 3.0, 0.0, 1.0))
    )
    components = {
        "oos": float(np.clip(oos_value, 0.0, 1.0)),
        "robustness": float(np.clip(robustness_value, 0.0, 1.0)),
        "stress": float(np.clip(stress_value, 0.0, 1.0)),
        "regime": float(np.clip(regime_value, 0.0, 1.0)),
        "discovery": float(np.clip(1.0 - q_value, 0.0, 1.0)),
        "drift": float(np.clip(drift_value, 0.0, 1.0)),
        "holdout": holdout_component,
    }
    score = float(
        np.clip(
            0.15 * components["oos"]
            + 0.15 * components["robustness"]
            + 0.15 * components["stress"]
            + 0.10 * components["regime"]
            + 0.15 * components["discovery"]
            + 0.10 * components["drift"]
            + 0.20 * components["holdout"],
            0.0,
            1.0,
        )
    )

    reasons: list[str] = []
    if not finite_evidence:
        reasons.append("non_finite_evidence")
    if oos_value < min_oos_coverage:
        reasons.append("oos_coverage")
    if robustness_value < min_robustness:
        reasons.append("robustness")
    if stress_value < min_stress_score:
        reasons.append("stress")
    if regime_value < min_regime_score:
        reasons.append("regime")
    if q_value > max_discovery_q_value:
        reasons.append("false_discovery_risk")
    if drift_degraded or drift_value < min_drift_score:
        reasons.append("recent_drift")
    if not holdout_passed or holdout_value is None or not holdout_finite:
        reasons.append("blind_holdout")

    return PaperReadinessReport(
        ready=not reasons,
        score=score,
        reasons=tuple(reasons),
        components=components,
    )
