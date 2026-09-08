from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from stockbot.data.schemas import DatasetMetadata
from stockbot.ml.models import ModelConfig
from stockbot.research.regime_eval import build_market_regime_series
from stockbot.research.regime_router import RegimeRouterReport, build_regime_router
from stockbot.research.regime_specialists import RegimeSpecialistResult, select_best_regime_specialists


@dataclass(frozen=True)
class RegimeSpecialistDiagnostics:
    specialists: dict[str, RegimeSpecialistResult]
    router_report: RegimeRouterReport | None
    candidate_pairs_tested: int


def select_specialist_candidate_pairs(
    candidates: list[Any] | tuple[Any, ...],
    *,
    top_k_per_horizon: int = 2,
) -> list[tuple[int, ModelConfig]]:
    """Select a small diverse set of strong generalists for regime specialization."""

    if top_k_per_horizon <= 0:
        raise ValueError("top_k_per_horizon must be positive")

    by_horizon: dict[int, list[Any]] = {}
    for candidate in candidates:
        if not getattr(candidate, "gate", None) or not candidate.gate.passed:
            continue
        horizon = int(candidate.horizon)
        by_horizon.setdefault(horizon, []).append(candidate)

    selected: list[tuple[int, ModelConfig]] = []
    for horizon in sorted(by_horizon):
        rows = sorted(
            by_horizon[horizon],
            key=lambda item: float(getattr(item, "selection_score", item.factory_score)),
            reverse=True,
        )[:top_k_per_horizon]
        for candidate in rows:
            selected.append(
                (
                    horizon,
                    ModelConfig(
                        candidate.model_name,
                        params=dict(candidate.model_params),
                        seed=int(candidate.seed),
                    ),
                )
            )
    return selected


def run_regime_specialist_diagnostics(
    bars: pd.DataFrame,
    metadata: DatasetMetadata,
    candidates: list[Any] | tuple[Any, ...],
    *,
    top_k_per_horizon: int = 2,
) -> RegimeSpecialistDiagnostics:
    """Train regime specialists from top generalist candidates and evaluate a router."""

    pairs = select_specialist_candidate_pairs(
        candidates,
        top_k_per_horizon=top_k_per_horizon,
    )
    if not pairs:
        return RegimeSpecialistDiagnostics({}, None, 0)

    specialists = select_best_regime_specialists(bars, metadata, pairs)
    regime_series = build_market_regime_series(bars)
    router = build_regime_router(specialists, regime_series) if specialists else None
    return RegimeSpecialistDiagnostics(
        specialists=specialists,
        router_report=router,
        candidate_pairs_tested=len(pairs),
    )
