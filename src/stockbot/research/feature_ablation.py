from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockbot.data.schemas import DatasetMetadata
from stockbot.ml.models import ModelConfig
from stockbot.research.training_pipeline import FEATURE_COLUMNS, run_training_research


DEFAULT_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "momentum_returns": (
        "return_1",
        "return_5",
        "return_20",
        "momentum_5",
        "momentum_20",
        "momentum_60",
    ),
    "volatility": (
        "realized_vol_5",
        "realized_vol_20",
        "realized_vol_60",
    ),
    "liquidity_volume": (
        "volume_z_5",
        "volume_z_20",
        "volume_z_60",
        "log_dollar_volume",
        "dollar_volume_z_20",
    ),
    "trend_location": (
        "sma_5_dist",
        "sma_20_dist",
        "sma_60_dist",
        "distance_high_20",
        "distance_low_20",
        "rsi_14",
    ),
    "microstructure": (
        "range_1",
        "range_20",
        "gap_1",
    ),
    "interactions": (
        "momentum_vol_ratio_20",
        "momentum_volume_interaction",
    ),
    "cross_sectional_ranks": (
        "return_1_rank",
        "momentum_5_rank",
        "momentum_rank",
        "momentum_60_rank",
        "volatility_5_rank",
        "volatility_rank",
        "volatility_60_rank",
        "volume_rank",
        "liquidity_rank",
        "range_rank",
        "rsi_rank",
    ),
}


@dataclass(frozen=True)
class FeatureAblationResult:
    group: str
    removed_features: tuple[str, ...]
    ablated_score: float
    score_impact: float
    ablated_sharpe: float
    ablated_cagr: float


@dataclass(frozen=True)
class FeatureAblationReport:
    baseline_score: float
    baseline_sharpe: float
    baseline_cagr: float
    results: tuple[FeatureAblationResult, ...]
    recommended_drop_groups: tuple[str, ...]


def evaluate_feature_group_ablation(
    bars: pd.DataFrame,
    metadata: DatasetMetadata,
    model_config: ModelConfig,
    *,
    horizon: int,
    groups: dict[str, tuple[str, ...]] | None = None,
    drop_improvement_threshold: float = 0.05,
    train_periods: int | None = None,
    test_periods: int | None = None,
) -> FeatureAblationReport:
    """Refit one challenger after removing each feature group from the research set.

    `score_impact = baseline - ablated`; positive impact means the removed group was
    useful, while a sufficiently negative impact marks the group as a candidate for
    removal in a later feature-subset experiment. No feature is automatically deleted.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if drop_improvement_threshold < 0:
        raise ValueError("drop_improvement_threshold must be non-negative")

    feature_groups = groups or DEFAULT_FEATURE_GROUPS
    baseline_run = run_training_research(
        bars,
        metadata,
        model_configs=(model_config,),
        horizon=horizon,
        max_workers=1,
        train_periods=train_periods,
        test_periods=test_periods,
        feature_columns=FEATURE_COLUMNS,
    )
    if not baseline_run.leaderboard:
        raise ValueError("baseline feature experiment produced no result")
    baseline = baseline_run.leaderboard[0]

    results: list[FeatureAblationResult] = []
    for group_name, features_to_remove in feature_groups.items():
        selected = tuple(feature for feature in FEATURE_COLUMNS if feature not in set(features_to_remove))
        if not selected:
            continue
        run = run_training_research(
            bars,
            metadata,
            model_configs=(model_config,),
            horizon=horizon,
            max_workers=1,
            train_periods=train_periods,
            test_periods=test_periods,
            feature_columns=selected,
        )
        if not run.leaderboard:
            continue
        ablated = run.leaderboard[0]
        results.append(
            FeatureAblationResult(
                group=str(group_name),
                removed_features=tuple(features_to_remove),
                ablated_score=float(ablated.score),
                score_impact=float(baseline.score - ablated.score),
                ablated_sharpe=float(ablated.metrics.get("sharpe", 0.0)),
                ablated_cagr=float(ablated.metrics.get("cagr", 0.0)),
            )
        )

    recommended = tuple(
        result.group
        for result in results
        if result.score_impact < -drop_improvement_threshold
    )
    return FeatureAblationReport(
        baseline_score=float(baseline.score),
        baseline_sharpe=float(baseline.metrics.get("sharpe", 0.0)),
        baseline_cagr=float(baseline.metrics.get("cagr", 0.0)),
        results=tuple(results),
        recommended_drop_groups=recommended,
    )
