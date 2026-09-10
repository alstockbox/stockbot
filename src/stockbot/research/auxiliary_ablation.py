from __future__ import annotations

from dataclasses import dataclass

from stockbot.data.point_in_time_features import PointInTimeFeatureStore
from stockbot.data.schemas import DatasetMetadata
from stockbot.ml.models import ModelConfig
from stockbot.research.training_pipeline import run_training_research


@dataclass(frozen=True)
class AuxiliaryAblationResult:
    feature_name: str
    baseline_score: float
    ablated_score: float
    score_impact: float
    ablated_sharpe: float
    ablated_cagr: float


@dataclass(frozen=True)
class AuxiliaryAblationReport:
    auxiliary_features: tuple[str, ...]
    baseline_score: float
    baseline_sharpe: float
    baseline_cagr: float
    no_auxiliary_score: float
    aggregate_score_impact: float
    results: tuple[AuxiliaryAblationResult, ...]
    useful_features: tuple[str, ...]
    harmful_features: tuple[str, ...]


def _raw_auxiliary_names(feature_columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        column.removeprefix("aux__")
        for column in feature_columns
        if column.startswith("aux__")
    )


def evaluate_auxiliary_feature_ablation(
    bars,
    metadata: DatasetMetadata,
    model_config: ModelConfig,
    *,
    horizon: int,
    feature_columns: tuple[str, ...] | list[str],
    auxiliary_store: PointInTimeFeatureStore,
    train_periods: int | None = None,
    test_periods: int | None = None,
    auxiliary_max_age_days: int | None = None,
    auxiliary_min_coverage: float = 0.80,
    useful_threshold: float = 0.05,
) -> AuxiliaryAblationReport:
    """Refit one candidate while removing each external point-in-time feature.

    This is a research-partition diagnostic. Positive `score_impact` means removing the
    feature reduced OOS score; negative values mean the feature hurt this candidate.
    The result must not be used as a substitute for independent holdout validation.
    """

    if useful_threshold < 0.0:
        raise ValueError("useful_threshold must be non-negative")
    exact_features = tuple(str(value) for value in feature_columns)
    auxiliary_names = _raw_auxiliary_names(exact_features)
    if not auxiliary_names:
        raise ValueError("auxiliary ablation requires at least one aux__ feature")
    missing = sorted(set(auxiliary_names).difference(auxiliary_store.feature_names))
    if missing:
        raise ValueError(f"auxiliary ablation store is missing features: {missing}")

    baseline_run = run_training_research(
        bars,
        metadata,
        model_configs=(model_config,),
        horizon=horizon,
        max_workers=1,
        train_periods=train_periods,
        test_periods=test_periods,
        feature_columns=exact_features,
        auxiliary_store=auxiliary_store,
        auxiliary_feature_names=auxiliary_names,
        auxiliary_max_age_days=auxiliary_max_age_days,
        auxiliary_min_coverage=auxiliary_min_coverage,
    )
    if not baseline_run.leaderboard:
        raise ValueError("auxiliary ablation baseline produced no result")
    baseline = baseline_run.leaderboard[0]

    base_features = tuple(column for column in exact_features if not column.startswith("aux__"))
    no_aux_run = run_training_research(
        bars,
        metadata,
        model_configs=(model_config,),
        horizon=horizon,
        max_workers=1,
        train_periods=train_periods,
        test_periods=test_periods,
        feature_columns=base_features,
    )
    if not no_aux_run.leaderboard:
        raise ValueError("no-auxiliary ablation produced no result")
    no_aux = no_aux_run.leaderboard[0]

    results: list[AuxiliaryAblationResult] = []
    for raw_name in auxiliary_names:
        removed_column = f"aux__{raw_name}"
        selected = tuple(column for column in exact_features if column != removed_column)
        remaining_auxiliary = _raw_auxiliary_names(selected)
        run = run_training_research(
            bars,
            metadata,
            model_configs=(model_config,),
            horizon=horizon,
            max_workers=1,
            train_periods=train_periods,
            test_periods=test_periods,
            feature_columns=selected,
            auxiliary_store=(auxiliary_store if remaining_auxiliary else None),
            auxiliary_feature_names=(remaining_auxiliary if remaining_auxiliary else None),
            auxiliary_max_age_days=auxiliary_max_age_days,
            auxiliary_min_coverage=auxiliary_min_coverage,
        )
        if not run.leaderboard:
            continue
        ablated = run.leaderboard[0]
        results.append(
            AuxiliaryAblationResult(
                feature_name=raw_name,
                baseline_score=float(baseline.score),
                ablated_score=float(ablated.score),
                score_impact=float(baseline.score - ablated.score),
                ablated_sharpe=float(ablated.metrics.get("sharpe", 0.0)),
                ablated_cagr=float(ablated.metrics.get("cagr", 0.0)),
            )
        )

    useful = tuple(
        row.feature_name for row in results if row.score_impact >= useful_threshold
    )
    harmful = tuple(
        row.feature_name for row in results if row.score_impact <= -useful_threshold
    )
    return AuxiliaryAblationReport(
        auxiliary_features=auxiliary_names,
        baseline_score=float(baseline.score),
        baseline_sharpe=float(baseline.metrics.get("sharpe", 0.0)),
        baseline_cagr=float(baseline.metrics.get("cagr", 0.0)),
        no_auxiliary_score=float(no_aux.score),
        aggregate_score_impact=float(baseline.score - no_aux.score),
        results=tuple(results),
        useful_features=useful,
        harmful_features=harmful,
    )
