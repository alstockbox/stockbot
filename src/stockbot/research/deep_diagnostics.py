from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from stockbot.data.schemas import DatasetMetadata
from stockbot.ml.models import ModelConfig
from stockbot.research.bootstrap_uncertainty import BootstrapUncertaintyReport, evaluate_block_bootstrap_uncertainty
from stockbot.research.capacity_curve import CapacityCurveReport, evaluate_capacity_curve
from stockbot.research.feature_ablation import FeatureAblationReport, evaluate_feature_group_ablation
from stockbot.research.liquidity_execution import LiquidityExecutionConfig, LiquidityExecutionReport, simulate_liquidity_aware_execution
from stockbot.research.policy_search import PolicyArenaReport, evaluate_policy_arena
from stockbot.research.window_robustness import WindowRobustnessReport, evaluate_training_window_robustness


@dataclass(frozen=True)
class CandidateDeepDiagnostics:
    experiment_id: str
    horizon: int
    model_name: str
    policy_arena: PolicyArenaReport
    bootstrap_uncertainty: BootstrapUncertaintyReport
    liquidity_execution: LiquidityExecutionReport
    capacity_curve: CapacityCurveReport
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
    liquidity_config: LiquidityExecutionConfig | None = None,
    capacity_levels: tuple[float, ...] = (
        10_000.0,
        25_000.0,
        50_000.0,
        100_000.0,
        250_000.0,
        500_000.0,
        1_000_000.0,
        2_500_000.0,
        5_000_000.0,
    ),
    bootstrap_block_size: int = 21,
    bootstrap_samples: int = 500,
    bootstrap_confidence_level: float = 0.90,
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
        best_policy = policy.best.policy
        bootstrap = evaluate_block_bootstrap_uncertainty(
            policy.best.net_returns,
            block_size=min(bootstrap_block_size, len(policy.best.net_returns)),
            samples=bootstrap_samples,
            confidence_level=bootstrap_confidence_level,
        )
        if candidate.result.predictions is None:
            raise ValueError("deep diagnostics require retained OOS predictions")
        liquidity = simulate_liquidity_aware_execution(
            research_bars,
            candidate.result.predictions,
            top_fraction=best_policy.top_fraction,
            weighting=best_policy.weighting,
            config=liquidity_config,
            oos_coverage=candidate.oos_coverage,
        )
        capacity_curve = evaluate_capacity_curve(
            research_bars,
            candidate.result.predictions,
            capital_levels=capacity_levels,
            top_fraction=best_policy.top_fraction,
            weighting=best_policy.weighting,
            base_config=liquidity_config,
            oos_coverage=candidate.oos_coverage,
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
            bootstrap_uncertainty=bootstrap,
            liquidity_execution=liquidity,
            capacity_curve=capacity_curve,
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
            "bootstrap_uncertainty": {
                "confidence_score": item.bootstrap_uncertainty.confidence_score,
                "probability_positive_cagr": item.bootstrap_uncertainty.probability_positive_cagr,
                "probability_positive_sharpe": item.bootstrap_uncertainty.probability_positive_sharpe,
                "samples": item.bootstrap_uncertainty.samples,
                "block_size": item.bootstrap_uncertainty.block_size,
                "confidence_level": item.bootstrap_uncertainty.confidence_level,
                "cagr": {
                    "lower": item.bootstrap_uncertainty.cagr.lower,
                    "median": item.bootstrap_uncertainty.cagr.median,
                    "upper": item.bootstrap_uncertainty.cagr.upper,
                },
                "sharpe": {
                    "lower": item.bootstrap_uncertainty.sharpe.lower,
                    "median": item.bootstrap_uncertainty.sharpe.median,
                    "upper": item.bootstrap_uncertainty.sharpe.upper,
                },
                "max_drawdown": {
                    "lower": item.bootstrap_uncertainty.max_drawdown.lower,
                    "median": item.bootstrap_uncertainty.max_drawdown.median,
                    "upper": item.bootstrap_uncertainty.max_drawdown.upper,
                },
            },
            "liquidity_execution": {
                "score": item.liquidity_execution.score,
                "partial_fill_fraction": item.liquidity_execution.partial_fill_fraction,
                "average_participation": item.liquidity_execution.average_participation,
                "max_participation": item.liquidity_execution.max_participation,
                "average_tracking_error": item.liquidity_execution.average_tracking_error,
                "capacity_utilization": item.liquidity_execution.capacity_utilization,
                "cost_ratio": item.liquidity_execution.metrics.get("cost_ratio", 0.0),
            },
            "capacity_curve": {
                "capacity_score": item.capacity_curve.capacity_score,
                "max_effective_capital": item.capacity_curve.max_effective_capital,
                "first_break_capital": item.capacity_curve.first_break_capital,
                "baseline_capital": item.capacity_curve.baseline_capital,
                "baseline_score": item.capacity_curve.baseline_score,
                "score_retention_at_max": item.capacity_curve.score_retention_at_max,
                "points": [
                    {
                        "capital": point.capital,
                        "score": point.score,
                        "cagr": point.cagr,
                        "sharpe": point.sharpe,
                        "max_drawdown": point.max_drawdown,
                        "partial_fill_fraction": point.partial_fill_fraction,
                        "average_tracking_error": point.average_tracking_error,
                        "average_participation": point.average_participation,
                        "cost_ratio": point.cost_ratio,
                        "score_retention": point.score_retention,
                        "usable": point.usable,
                    }
                    for point in item.capacity_curve.points
                ],
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
