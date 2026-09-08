from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockbot.data.point_in_time_features import AuxiliaryFeatureCoverageReport, PointInTimeFeatureStore
from stockbot.features.cross_sectional import add_cross_sectional_features


BASE_RESEARCH_FEATURE_COLUMNS = (
    "return_1",
    "return_5",
    "return_20",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "realized_vol_5",
    "realized_vol_20",
    "realized_vol_60",
    "volume_z_5",
    "volume_z_20",
    "volume_z_60",
    "log_dollar_volume",
    "dollar_volume_z_20",
    "sma_5_dist",
    "sma_20_dist",
    "sma_60_dist",
    "distance_high_20",
    "distance_low_20",
    "range_1",
    "range_20",
    "gap_1",
    "rsi_14",
    "momentum_vol_ratio_20",
    "momentum_volume_interaction",
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
)


@dataclass(frozen=True)
class ResearchFeatureMatrix:
    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    auxiliary_features: tuple[str, ...]
    auxiliary_coverage: AuxiliaryFeatureCoverageReport | None


def build_research_feature_matrix(
    panel: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str] | None = None,
    auxiliary_store: PointInTimeFeatureStore | None = None,
    auxiliary_feature_names: tuple[str, ...] | list[str] | None = None,
    auxiliary_max_age_days: int | None = None,
    auxiliary_min_coverage: float = 0.80,
) -> ResearchFeatureMatrix:
    """Build one reproducible causal matrix used by training and replay evaluation.

    External features can enter only through `PointInTimeFeatureStore`, which enforces
    observation/availability semantics and revision awareness. They are namespaced as
    `aux__*` so artifacts can replay the exact same feature set later.
    """

    if not 0.0 < auxiliary_min_coverage <= 1.0:
        raise ValueError("auxiliary_min_coverage must be in (0,1]")
    if auxiliary_store is None and auxiliary_feature_names is not None:
        raise ValueError("auxiliary_feature_names require a PointInTimeFeatureStore")

    all_features = add_cross_sectional_features(panel)
    auxiliary_columns: tuple[str, ...] = ()
    coverage: AuxiliaryFeatureCoverageReport | None = None
    if auxiliary_store is not None:
        requested = (
            auxiliary_store.feature_names
            if auxiliary_feature_names is None
            else tuple(str(value).strip() for value in auxiliary_feature_names)
        )
        coverage = auxiliary_store.coverage_for_index(
            panel.index,
            feature_names=requested,
            max_age_days=auxiliary_max_age_days,
            min_coverage=auxiliary_min_coverage,
        )
        if not coverage.research_grade_auxiliary:
            raise ValueError(
                "auxiliary point-in-time feature gate failed: " + ",".join(coverage.reasons)
            )
        auxiliary = auxiliary_store.materialize(
            panel.index,
            feature_names=requested,
            max_age_days=auxiliary_max_age_days,
        )
        auxiliary = auxiliary.rename(columns={column: f"aux__{column}" for column in auxiliary.columns})
        collisions = sorted(set(auxiliary.columns).intersection(all_features.columns))
        if collisions:
            raise ValueError(f"auxiliary feature name collision: {collisions}")
        all_features = pd.concat([all_features, auxiliary], axis=1)
        auxiliary_columns = tuple(auxiliary.columns)

    selected = (
        tuple(feature_columns)
        if feature_columns is not None
        else BASE_RESEARCH_FEATURE_COLUMNS + auxiliary_columns
    )
    if not selected:
        raise ValueError("at least one feature column is required")
    missing = sorted(set(selected).difference(all_features.columns))
    if missing:
        raise ValueError(f"unknown feature columns: {missing}")
    return ResearchFeatureMatrix(
        frame=all_features.loc[:, selected],
        feature_names=selected,
        auxiliary_features=auxiliary_columns,
        auxiliary_coverage=coverage,
    )
