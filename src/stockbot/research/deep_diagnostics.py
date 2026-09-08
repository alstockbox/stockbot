from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from stockbot.data.point_in_time_features import PointInTimeFeatureStore
from stockbot.data.schemas import DatasetMetadata
from stockbot.data.universe import PointInTimeUniverse
from stockbot.ml.models import ModelConfig
from stockbot.research.auxiliary_ablation import (
    AuxiliaryAblationReport,
    evaluate_auxiliary_feature_ablation,
)
from stockbot.research.bootstrap_uncertainty import BootstrapUncertaintyReport, evaluate_block_bootstrap_uncertainty
from stockbot.research.capacity_curve import CapacityCurveReport, evaluate_capacity_curve
from stockbot.research.factor_exposure import FactorExposureReport, build_internal_factor_returns, evaluate_factor_exposure
from stockbot.research.feature_ablation import FeatureAblationReport, evaluate_feature_group_ablation
from stockbot.research.liquidity_execution import LiquidityExecutionConfig, LiquidityExecutionReport, simulate_liquidity_aware_execution
from stockbot.research.neutralization import NeutralizationReport, evaluate_sector_factor_neutralization
from stockbot.research.policy_search import PolicyArenaReport, evaluate_policy_arena
from stockbot.research.stacking import StackingReport, evaluate_oos_stacking
from stockbot.research.window_robustness import WindowRobustnessReport, evaluate_training_window_robustness


@dataclass(frozen=True)
class CandidateDeepDiagnostics:
    experiment_id: str
    horizon: int
    model_name: str
    policy_arena: PolicyArenaReport
    bootstrap_uncertainty: BootstrapUncertaintyReport
    factor_exposure: FactorExposureReport
    liquidity_execution: LiquidityExecutionReport
    capacity_curve: CapacityCurveReport
    window_robustness: WindowRobustnessReport
    feature_ablation: FeatureAblationReport
    neutralization: NeutralizationReport | None = None
    auxiliary_ablation: AuxiliaryAblationReport | None = None


@dataclass(frozen=True)
class DeepResearchDiagnostics:
    candidates: dict[str, CandidateDeepDiagnostics]
    stacking_reports: dict[int, StackingReport]
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


def _stacking_groups(
    candidates: list[Any] | tuple[Any, ...],
    *,
    top_k_per_horizon: int,
) -> dict[int, list[Any]]:
    if top_k_per_horizon < 2:
        raise ValueError("stacking top-k must be at least 2")
    grouped: dict[int, list[Any]] = {}
    for candidate in candidates:
        if not getattr(candidate, "gate", None) or not candidate.gate.passed:
            continue
        predictions = getattr(candidate.result, "predictions", None)
        if predictions is None or len(predictions.dropna()) == 0:
            continue
        grouped.setdefault(int(candidate.horizon), []).append(candidate)

    output: dict[int, list[Any]] = {}
    for horizon, rows in grouped.items():
        ranked = sorted(
            rows,
            key=lambda item: float(getattr(item, "selection_score", item.factory_score)),
            reverse=True,
        )
        if len(ranked) >= 2:
            output[horizon] = ranked[:top_k_per_horizon]
    return output


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
    stacking_top_k_per_horizon: int = 3,
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
    point_in_time_universe: PointInTimeUniverse | None = None,
    auxiliary_store: PointInTimeFeatureStore | None = None,
    auxiliary_feature_names: tuple[str, ...] | list[str] | None = None,
    auxiliary_max_age_days: int | None = None,
    auxiliary_min_coverage: float = 0.80,
) -> DeepResearchDiagnostics:
    """Run expensive diagnostics only on top generalists and only before blind holdout."""

    research_bars = _research_partition(bars, getattr(factory_report, "holdout_start", None))
    factor_returns = build_internal_factor_returns(research_bars)
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
        factor_exposure = evaluate_factor_exposure(policy.best.net_returns, factor_returns)
        if candidate.result.predictions is None:
            raise ValueError("deep diagnostics require retained OOS predictions")

        artifact_feature_names = tuple(
            str(value) for value in getattr(candidate.result.artifact, "feature_names", ())
        )
        uses_auxiliary = any(name.startswith("aux__") for name in artifact_feature_names)
        if uses_auxiliary and auxiliary_store is None:
            raise ValueError(
                "deep diagnostics cannot replay an auxiliary-feature candidate without its PointInTimeFeatureStore"
            )
        replay_store = auxiliary_store if uses_auxiliary else None
        replay_aux_names = auxiliary_feature_names if uses_auxiliary else None
        replay_feature_names = artifact_feature_names or None

        neutralization = None
        if point_in_time_universe is not None:
            neutralization = evaluate_sector_factor_neutralization(
                research_bars,
                candidate.result.predictions,
                point_in_time_universe,
                top_fraction=best_policy.top_fraction,
                weighting=best_policy.weighting,
            )

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
            feature_columns=replay_feature_names,
            auxiliary_store=replay_store,
            auxiliary_feature_names=replay_aux_names,
            auxiliary_max_age_days=auxiliary_max_age_days,
            auxiliary_min_coverage=auxiliary_min_coverage,
        )
        ablation = evaluate_feature_group_ablation(
            research_bars,
            metadata,
            model,
            horizon=int(candidate.horizon),
            train_periods=(None if not windows.results else windows.results[0].train_periods),
            test_periods=test_periods,
            feature_columns=replay_feature_names,
            auxiliary_store=replay_store,
            auxiliary_feature_names=replay_aux_names,
            auxiliary_max_age_days=auxiliary_max_age_days,
            auxiliary_min_coverage=auxiliary_min_coverage,
        )

        auxiliary_ablation = None
        if uses_auxiliary and replay_store is not None and replay_feature_names is not None:
            auxiliary_ablation = evaluate_auxiliary_feature_ablation(
                research_bars,
                metadata,
                model,
                horizon=int(candidate.horizon),
                feature_columns=replay_feature_names,
                auxiliary_store=replay_store,
                train_periods=(None if not windows.results else windows.results[0].train_periods),
                test_periods=test_periods,
                auxiliary_max_age_days=auxiliary_max_age_days,
                auxiliary_min_coverage=auxiliary_min_coverage,
            )

        diagnostics[candidate.experiment_id] = CandidateDeepDiagnostics(
            experiment_id=candidate.experiment_id,
            horizon=int(candidate.horizon),
            model_name=candidate.model_name,
            policy_arena=policy,
            bootstrap_uncertainty=bootstrap,
            factor_exposure=factor_exposure,
            liquidity_execution=liquidity,
            capacity_curve=capacity_curve,
            window_robustness=windows,
            feature_ablation=ablation,
            neutralization=neutralization,
            auxiliary_ablation=auxiliary_ablation,
        )

    stacking_reports: dict[int, StackingReport] = {}
    for horizon, members in _stacking_groups(
        factory_report.candidates,
        top_k_per_horizon=stacking_top_k_per_horizon,
    ).items():
        stacking_reports[horizon] = evaluate_oos_stacking(
            research_bars,
            members,
            horizon=horizon,
        )

    return DeepResearchDiagnostics(
        candidates=diagnostics,
        stacking_reports=stacking_reports,
        candidate_count=len(diagnostics),
    )


def compact_deep_diagnostics(diagnostics: DeepResearchDiagnostics) -> dict[str, object]:
    """Produce a JSON-friendly summary without serializing prediction/return series."""

    output: dict[str, object] = {
        "candidate_count": diagnostics.candidate_count,
        "candidates": {},
        "stacking": {
            str(horizon): {
                "member_ids": list(report.member_ids),
                "meta_model": report.meta_model.name,
                "score": report.score,
                "oos_coverage": report.oos_coverage,
                "robustness": report.robustness,
                "sharpe": report.metrics.get("sharpe", 0.0),
                "cagr": report.metrics.get("cagr", 0.0),
                "max_drawdown": report.metrics.get("max_drawdown", 0.0),
                "stress_score": report.stress_report.score,
            }
            for horizon, report in diagnostics.stacking_reports.items()
        },
    }
    rows: dict[str, object] = {}
    for experiment_id, item in diagnostics.candidates.items():
        baseline_score = None if item.policy_arena.baseline is None else item.policy_arena.baseline.score
        neutralization = None
        if item.neutralization is not None:
            neutralization = {
                "baseline_score": item.neutralization.baseline_score,
                "neutralized_score": item.neutralization.neutralized_score,
                "score_delta": item.neutralization.score_delta,
                "sector_coverage": item.neutralization.sector_coverage,
                "prediction_coverage": item.neutralization.prediction_coverage,
                "sector_mean_before": item.neutralization.average_abs_sector_mean_before,
                "sector_mean_after": item.neutralization.average_abs_sector_mean_after,
                "factor_correlations_before": dict(item.neutralization.factor_correlations_before),
                "factor_correlations_after": dict(item.neutralization.factor_correlations_after),
                "neutralized_sharpe": item.neutralization.neutralized_metrics.get("sharpe", 0.0),
                "neutralized_cagr": item.neutralization.neutralized_metrics.get("cagr", 0.0),
                "neutralized_stress": item.neutralization.neutralized_stress.score,
            }

        auxiliary_ablation = None
        if item.auxiliary_ablation is not None:
            auxiliary_ablation = {
                "auxiliary_features": list(item.auxiliary_ablation.auxiliary_features),
                "baseline_score": item.auxiliary_ablation.baseline_score,
                "no_auxiliary_score": item.auxiliary_ablation.no_auxiliary_score,
                "aggregate_score_impact": item.auxiliary_ablation.aggregate_score_impact,
                "useful_features": list(item.auxiliary_ablation.useful_features),
                "harmful_features": list(item.auxiliary_ablation.harmful_features),
                "feature_impacts": {
                    row.feature_name: row.score_impact
                    for row in item.auxiliary_ablation.results
                },
            }

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
            "factor_exposure": {
                "factor_betas": dict(item.factor_exposure.factor_betas),
                "daily_alpha": item.factor_exposure.daily_alpha,
                "annualized_alpha": item.factor_exposure.annualized_alpha,
                "r_squared": item.factor_exposure.r_squared,
                "explained_fraction": item.factor_exposure.explained_fraction,
                "idiosyncratic_score": item.factor_exposure.idiosyncratic_score,
                "residual_sharpe": item.factor_exposure.residual_metrics.get("sharpe", 0.0),
                "residual_cagr": item.factor_exposure.residual_metrics.get("cagr", 0.0),
                "observations": item.factor_exposure.observations,
            },
            "neutralization": neutralization,
            "auxiliary_ablation": auxiliary_ablation,
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
