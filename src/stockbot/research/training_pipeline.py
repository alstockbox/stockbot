from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pandas as pd

from stockbot.arena.experiments import ExperimentConfig, ModelExperimentResult, run_model_experiment
from stockbot.arena.leaderboard import eligible_for_promotion, rank_experiments
from stockbot.data.panel import build_panel
from stockbot.data.point_in_time_features import PointInTimeFeatureStore
from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.features.research_matrix import BASE_RESEARCH_FEATURE_COLUMNS, build_research_feature_matrix
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

# Backward-compatible public alias used by holdout/ablation modules.
FEATURE_COLUMNS = BASE_RESEARCH_FEATURE_COLUMNS


@dataclass(frozen=True)
class TrainingRun:
    leaderboard: list[ModelExperimentResult]
    champion_candidate: ModelExperimentResult | None
    data_grade: DataGrade
    dataset_fingerprint: str
    horizon: int
    auxiliary_features: tuple[str, ...] = ()


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
    auxiliary_store: PointInTimeFeatureStore | None = None,
    auxiliary_feature_names: tuple[str, ...] | list[str] | None = None,
    auxiliary_max_age_days: int | None = None,
    auxiliary_min_coverage: float = 0.80,
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
    matrix = build_research_feature_matrix(
        panel,
        feature_columns=feature_columns,
        auxiliary_store=auxiliary_store,
        auxiliary_feature_names=auxiliary_feature_names,
        auxiliary_max_age_days=auxiliary_max_age_days,
        auxiliary_min_coverage=auxiliary_min_coverage,
    )
    features = matrix.frame
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
    return TrainingRun(
        leaderboard,
        champion,
        metadata.grade,
        fingerprint,
        horizon,
        auxiliary_features=matrix.auxiliary_features,
    )
