from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pandas as pd

from stockbot.arena.experiments import ExperimentConfig, ModelExperimentResult, run_model_experiment
from stockbot.arena.leaderboard import eligible_for_promotion, rank_experiments
from stockbot.data.panel import build_panel
from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.features.cross_sectional import add_cross_sectional_features
from stockbot.ml.labels import make_panel_labels
from stockbot.ml.models import ModelConfig
from stockbot.ml.purged_cv import PurgedWalkForwardSplitter


DEFAULT_MODELS = (
    ModelConfig("ridge", seed=7),
    ModelConfig("elastic_net", seed=7),
    ModelConfig("extra_trees", seed=7),
    ModelConfig("random_forest", seed=7),
    ModelConfig("hist_gb", seed=7),
)

# Causal features spanning momentum, volatility, liquidity, trend/location,
# intraday structure, interactions and same-date cross-sectional ranks.
FEATURE_COLUMNS = (
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
class TrainingRun:
    leaderboard: list[ModelExperimentResult]
    champion_candidate: ModelExperimentResult | None
    data_grade: DataGrade
    dataset_fingerprint: str
    horizon: int


def run_training_research(
    bars: pd.DataFrame,
    metadata: DatasetMetadata,
    model_configs: list[ModelConfig] | tuple[ModelConfig, ...] | None = None,
    horizon: int = 5,
    *,
    max_workers: int = 1,
    train_periods: int | None = None,
    test_periods: int | None = None,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> TrainingRun:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if train_periods is not None and train_periods <= 0:
        raise ValueError("train_periods must be positive")
    if test_periods is not None and test_periods <= 0:
        raise ValueError("test_periods must be positive")

    panel = build_panel(bars)
    all_features = add_cross_sectional_features(panel)
    selected_features = tuple(feature_columns) if feature_columns is not None else FEATURE_COLUMNS
    if not selected_features:
        raise ValueError("at least one feature column is required")
    missing_features = sorted(set(selected_features).difference(all_features.columns))
    if missing_features:
        raise ValueError(f"unknown feature columns: {missing_features}")
    features = all_features.loc[:, selected_features]
    labels = make_panel_labels(panel, horizons=(horizon,))[f"fwd_return_{horizon}"]
    labels.name = f"fwd_return_{horizon}"

    n_dates = len(panel.index.get_level_values("timestamp").unique())
    effective_train = train_periods if train_periods is not None else max(60, min(126, n_dates // 2))
    effective_test = test_periods if test_periods is not None else max(10, min(21, n_dates // 10))
    if effective_train + horizon + effective_test > n_dates:
        raise ValueError("insufficient dates for requested walk-forward train/test periods")
    splitter = PurgedWalkForwardSplitter(effective_train, effective_test, horizon, horizon)
    configs = tuple(model_configs) if model_configs is not None else DEFAULT_MODELS

    def run_one(config: ModelConfig) -> ModelExperimentResult:
        return run_model_experiment(
            panel,
            features,
            labels,
            splitter,
            ExperimentConfig(config),
            metadata,
        )

    if max_workers == 1 or len(configs) <= 1:
        results = [run_one(config) for config in configs]
    else:
        workers = min(max_workers, len(configs))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stockbot-research") as executor:
            results = list(executor.map(run_one, configs))

    leaderboard = rank_experiments(results)
    champion = next((row for row in leaderboard if eligible_for_promotion(row, metadata)), None)
    fingerprint = leaderboard[0].artifact.dataset_fingerprint if leaderboard else ""
    return TrainingRun(leaderboard, champion, metadata.grade, fingerprint, horizon)
