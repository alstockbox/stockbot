from __future__ import annotations

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

DEFAULT_MODELS = (ModelConfig("ridge", seed=7),ModelConfig("elastic_net", seed=7),ModelConfig("extra_trees", seed=7),ModelConfig("random_forest", seed=7),ModelConfig("hist_gb", seed=7))
FEATURE_COLUMNS = ("return_1","momentum_20","realized_vol_20","volume_z_20","momentum_rank","volatility_rank","volume_rank")

@dataclass(frozen=True)
class TrainingRun:
    leaderboard: list[ModelExperimentResult]
    champion_candidate: ModelExperimentResult | None
    data_grade: DataGrade
    dataset_fingerprint: str
    horizon: int


def run_training_research(bars: pd.DataFrame, metadata: DatasetMetadata, model_configs: list[ModelConfig] | tuple[ModelConfig, ...] | None = None, horizon: int = 5) -> TrainingRun:
    if horizon <= 0: raise ValueError("horizon must be positive")
    panel = build_panel(bars); features = add_cross_sectional_features(panel).loc[:, FEATURE_COLUMNS]; labels = make_panel_labels(panel, horizons=(horizon,))[f"fwd_return_{horizon}"]; labels.name = f"fwd_return_{horizon}"
    n_dates = len(panel.index.get_level_values("timestamp").unique()); train_periods = max(60, min(126, n_dates // 2)); test_periods = max(10, min(21, n_dates // 10))
    splitter = PurgedWalkForwardSplitter(train_periods, test_periods, horizon, horizon); configs = tuple(model_configs) if model_configs is not None else DEFAULT_MODELS
    leaderboard = rank_experiments([run_model_experiment(panel, features, labels, splitter, ExperimentConfig(config), metadata) for config in configs])
    champion = next((row for row in leaderboard if eligible_for_promotion(row, metadata)), None); fingerprint = leaderboard[0].artifact.dataset_fingerprint if leaderboard else ""
    return TrainingRun(leaderboard, champion, metadata.grade, fingerprint, horizon)
