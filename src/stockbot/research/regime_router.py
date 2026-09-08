from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockbot.domain.models import MarketRegime
from stockbot.evaluation.metrics import performance_metrics
from stockbot.research.regime_specialists import RegimeSpecialistResult
from stockbot.research.stress import StressReport, evaluate_stress_suite


@dataclass(frozen=True)
class RegimeRouterReport:
    returns: pd.Series
    metrics: dict[str, float]
    stress_report: StressReport
    specialist_scores: dict[str, float]
    regime_observations: dict[str, int]
    score: float


def build_regime_router(
    specialists: dict[str, RegimeSpecialistResult],
    regime_series: pd.Series,
) -> RegimeRouterReport:
    """Route each date to the OOS return stream of the specialist for that regime.

    This is a research diagnostic, not an execution engine. It uses already-produced
    OOS specialist returns and a causal regime label; it does not refit models or send
    orders. Missing specialist observations remain flat rather than being backfilled
    from a future observation.
    """

    if not specialists:
        raise ValueError("at least one regime specialist is required")

    regimes = pd.Series(regime_series, dtype="object").sort_index()
    routed = pd.Series(0.0, index=regimes.index, dtype=float, name="regime_router_return")
    observations: dict[str, int] = {regime.value: 0 for regime in MarketRegime}
    specialist_scores: dict[str, float] = {}

    for regime_name, specialist in specialists.items():
        if regime_name not in {regime.value for regime in MarketRegime}:
            raise ValueError(f"unknown regime specialist key: {regime_name}")
        specialist_scores[regime_name] = float(specialist.score)
        active = regimes.eq(regime_name)
        observations[regime_name] = int(active.sum())
        specialist_returns = pd.Series(specialist.net_returns, dtype=float).reindex(regimes.index)
        routed.loc[active] = specialist_returns.loc[active].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    metrics = performance_metrics(routed)
    stress = evaluate_stress_suite(routed)
    sharpe = float(metrics.get("sharpe", 0.0))
    cagr = float(metrics.get("cagr", 0.0))
    drawdown = max(0.0, float(metrics.get("max_drawdown", 1.0)))
    score = float(
        np.clip(
            0.30 * np.clip((sharpe + 1.0) / 3.0, 0.0, 1.0)
            + 0.25 * np.clip(0.5 + cagr, 0.0, 1.0)
            + 0.20 * np.clip(1.0 - drawdown, 0.0, 1.0)
            + 0.25 * stress.score,
            0.0,
            1.0,
        )
    )

    return RegimeRouterReport(
        returns=routed,
        metrics={key: float(value) for key, value in metrics.items()},
        stress_report=stress,
        specialist_scores=specialist_scores,
        regime_observations=observations,
        score=score,
    )
