from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from stockbot.data.schemas import DatasetMetadata
from stockbot.ml.models import ModelConfig
from stockbot.research.feature_ablation import FeatureAblationReport, evaluate_feature_group_ablation
from stockbot.research.policy_search import PolicyArenaReport, evaluate_policy_arena
from stockbot.research.window_robustness import WindowRobustnessReport, evaluate_training_window_robustness


@dataclass(frozen=True)
class CandidateDeepDiagnostics:
    experiment_id: str
    horizon: int
    model_name: str
    policy_arena: PolicyArenaReport
    window_robustness: WindowRobustnessReport
    feature_ablation: FeatureAblationReport


@dataclass(frozen=True)
class DeepResearchDiagnostics:
    candidates: dict[str, CandidateDeepDiagnostics]
    candidate_count: int


def select_deep_diagnostic_candidates(
    candidates: list[Any] | tuple[Any, ...],
    *,
    top_k_per_horizon: int = 1,
) -> list[Any]:
    if top_k_per_horizon <= 0:
        raise ValueError("top_k_per_horizon must be positive")
    grouped: dict[int, list[Any]] = {}
    for candidate in candidates:
        if not getattr(candidate, "gate", None) or not candidate.gate.passed:
            continue
        grouped.setdefault(int(candidate.horizon), []).append(candidate)

    selected: list[Any] = []
    for horizon in sorted(grouped):
        rows = sorted(
            grouped[horizon],
            key=lambda item: float(getattr(item, "selection_score", item.factory_score)),
            reverse=True,
        )
        selected.extend(rows[:top_k_per_horizon])
    return selected


def _research_partition(bars: pd.DataFrame, holdout_start: pd.Timestamp | None) -> pd.DataFrame:
    if holdout_start is None:
        return bars.copy()
    timestamps = pd.to_datetime(bars["timestamp"], utc=True)
    cutoff = pd.Timestamp(holdout_start)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    return bars.loc[timestamps < cutoff].copy()


def run_deep_research_diagnostics(
    bars: pd.DataFrame,
    metadata: DatasetMetadata,
    factory_report: Any,
    *,
    top_k_per_horizon: int = 1,
    train_windows: tuple[int, ...] = (126, 252, 504),
    test_periods: int = 21,
) -> DeepResearchDiagnostics:
    """Run expensive diagnostics only on top generalists and only before blind holdout."""

    research_bars = _research_partition(bars, getattr(factory_report, "holdout_start", None))
    selected = select_deep_diagnostic_candidates(
        factory_report.candidates,
        top_k_per_horizon=top_k_per_horizon,
    )
    diagnostics: dict[str, CandidateDeepDiagnostics] = {}

    for candidate in selected:
        model = ModelConfig(
            candidate.model_name,
            params=dict(candidate.model_params),
            seed=int(candidate.seed),
        )
        policy = evaluate_policy_arena(
            research_bars,
            candidate.result,
        )
        windows = evaluate_training_window_robustness(
            research_bars,
            metadata,
            model,
            horizon=int(candidate.horizon),
            train_windows=train_windows,
            test_periods=test_periods,
        )
        ablation = evaluate_feature_group_ablation(
            research_bars,
            metadata,
            model,
            horizon=int(candidate.horizon),
            train_periods=(None if not windows.results else windows.results[0].train_periods),
            test_periods=test_periods,
        )
        diagnostics[candidate.experiment_id] = CandidateDeepDiagnostics(
            experiment_id=candidate.experiment_id,
            horizon=int(candidate.horizon),
            model_name=candidate.model_name,
            policy_arena=policy,
            window_robustness=windows,
            feature_ablation=ablation,
        )

    return DeepResearchDiagnostics(
        candidates=diagnostics,
        candidate_count=len(diagnostics),
    )


def compact_deep_diagnostics(diagnostics: DeepResearchDiagnostics) -> dict[str, object]:
    """Produce a JSON-friendly summary without serializing prediction/return series."""

    output: dict[str, object] = {"candidate_count": diagnostics.candidate_count, "candidates": {}}
    rows: dict[str, object] = {}
    for experiment_id, item in diagnostics.candidates.items():
        baseline_score = None if item.policy_arena.baseline is None else item.policy_arena.baseline.score
        rows[experiment_id] = {
            "horizon": item.horizon,
            "model_name": item.model_name,
            "best_policy": {
                "top_fraction": item.policy_arena.best.policy.top_fraction,
                "weighting": item.policy_arena.best.policy.weighting,
                "score": item.policy_arena.best.score,
                "baseline_score": baseline_score,
            },
            "window_robustness": {
                "score": item.window_robustness.score,
                "worst_score": item.window_robustness.worst_score,
                "positive_fraction": item.window_robustness.positive_fraction,
                "windows": [row.train_periods for row in item.window_robustness.results],
            },
            "feature_ablation": {
                "baseline_score": item.feature_ablation.baseline_score,
                "recommended_drop_groups": list(item.feature_ablation.recommended_drop_groups),
                "group_impacts": {
                    row.group: row.score_impact for row in item.feature_ablation.results
                },
            },
        }
    output["candidates"] = rows
    return output
