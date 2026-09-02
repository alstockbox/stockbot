from __future__ import annotations

from dataclasses import dataclass, replace
import math
import numpy as np
import pandas as pd

from stockbot.arena.scoring import research_score
from stockbot.costs.model import LinearCostModel
from stockbot.data.schemas import DatasetMetadata
from stockbot.evaluation.metrics import performance_metrics
from stockbot.ml.artifacts import ExperimentArtifact
from stockbot.ml.models import ModelConfig
from stockbot.ml.purged_cv import PurgedWalkForwardSplitter
from stockbot.ml.trainer import train_oos


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    top_fraction: float = 0.30
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    def __post_init__(self) -> None:
        if not 0 < self.top_fraction <= 1: raise ValueError("top_fraction must be in (0,1]")


@dataclass(frozen=True)
class ModelExperimentResult:
    name: str
    score: float
    metrics: dict[str, float]
    robustness: float
    oos_coverage: float
    artifact: ExperimentArtifact


def _signal_weights(predictions: pd.Series, top_fraction: float) -> pd.DataFrame:
    matrix = predictions.unstack("symbol").sort_index(); weights = pd.DataFrame(0.0, index=matrix.index, columns=matrix.columns)
    for dt, row in matrix.iterrows():
        valid = row.dropna().sort_values(ascending=False); positive = valid[valid > 0]
        if positive.empty: continue
        n = max(1, int(math.ceil(len(valid) * top_fraction))); selected = positive.iloc[:n].index
        weights.loc[dt, selected] = 1.0 / len(selected)
    return weights


def _evaluate_panel_predictions(panel: pd.DataFrame, predictions: pd.Series, config: ExperimentConfig) -> tuple[dict[str, float], float]:
    close = panel["close"].unstack("symbol").sort_index().astype(float)
    signal_weights = _signal_weights(predictions, config.top_fraction).reindex(close.index).fillna(0.0)
    executed = signal_weights.shift(1).fillna(0.0); asset_returns = close.pct_change().fillna(0.0); gross = (executed * asset_returns).sum(axis=1)
    turnover = executed.diff().abs().sum(axis=1)
    if len(turnover): turnover.iloc[0] = executed.iloc[0].abs().sum()
    cost_model = LinearCostModel(config.commission_bps, config.slippage_bps); cost_rate = turnover.apply(cost_model.rate_for_turnover); net = gross - cost_rate
    pred_dates = predictions.dropna().index.get_level_values("timestamp")
    if len(pred_dates):
        start = pred_dates.min(); net = net.loc[net.index >= start]; turnover = turnover.loc[turnover.index >= start]; executed = executed.loc[executed.index >= start]; cost_rate = cost_rate.loc[cost_rate.index >= start]
    metrics = performance_metrics(net); metrics["turnover"] = float(turnover.sum()); metrics["cost_ratio"] = float(cost_rate.sum()); metrics["concentration"] = float(executed.pow(2).sum(axis=1).mean()) if len(executed) else 0.0
    monthly = (1.0 + net).resample("21B").prod() - 1.0 if len(net) else pd.Series(dtype=float); metrics["instability"] = max(0.0, min(1.0, float(monthly.std(ddof=0)) if len(monthly) > 1 else 0.0))
    robustness = float(np.clip((1.0 - metrics["max_drawdown"]) * (1.0 - metrics["instability"]), 0.0, 1.0)); return metrics, robustness


def run_model_experiment(panel: pd.DataFrame, features: pd.DataFrame, labels: pd.Series, splitter: PurgedWalkForwardSplitter, config: ExperimentConfig, metadata: DatasetMetadata) -> ModelExperimentResult:
    label_name = labels.name or "target"; oos = train_oos(features, labels, splitter, config.model, metadata, label_name=label_name); metrics, robustness = _evaluate_panel_predictions(panel, oos.predictions, config)
    score = research_score(metrics, robustness * oos.artifact.oos_coverage); artifact = replace(oos.artifact, metrics=dict(metrics))
    return ModelExperimentResult(config.model.name, float(score), metrics, robustness, oos.artifact.oos_coverage, artifact)
