from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from stockbot.arena.experiments import ExperimentConfig, _evaluate_panel_predictions
from stockbot.data.panel import build_panel
from stockbot.features.cross_sectional import add_cross_sectional_features
from stockbot.ml.labels import make_panel_labels
from stockbot.ml.models import ModelConfig, build_model
from stockbot.research.objective import ObjectiveWeights, risk_adjusted_objective
from stockbot.research.stress import StressReport, evaluate_stress_suite
from stockbot.research.training_pipeline import FEATURE_COLUMNS


@dataclass(frozen=True)
class HoldoutConfig:
    fraction: float = 0.15
    min_holdout_periods: int = 42
    min_research_periods: int = 126
    min_prediction_coverage: float = 0.60
    max_drawdown: float = 0.30
    min_stress_score: float = 0.40
    min_score: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction < 0.5:
            raise ValueError("holdout fraction must be in (0, 0.5)")
        if self.min_holdout_periods <= 0 or self.min_research_periods <= 0:
            raise ValueError("holdout/research periods must be positive")


@dataclass(frozen=True)
class HoldoutReport:
    holdout_start: pd.Timestamp
    periods: int
    metrics: dict[str, float]
    robustness: float
    prediction_coverage: float
    stress_score: float
    stress_report: StressReport
    score: float
    passed: bool
    reasons: tuple[str, ...]


def _unique_dates(bars: pd.DataFrame) -> pd.DatetimeIndex:
    if "timestamp" not in bars.columns:
        raise ValueError("bars must contain timestamp")
    dates = pd.DatetimeIndex(pd.to_datetime(bars["timestamp"], utc=True).unique()).sort_values()
    if dates.empty:
        raise ValueError("bars contain no timestamps")
    return dates


def split_research_holdout(
    bars: pd.DataFrame,
    config: HoldoutConfig | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Reserve an untouched final time block before model-family/hyperparameter search."""

    cfg = config or HoldoutConfig()
    dates = _unique_dates(bars)
    desired = max(cfg.min_holdout_periods, int(math.ceil(len(dates) * cfg.fraction)))
    max_holdout = len(dates) - cfg.min_research_periods
    if max_holdout < cfg.min_holdout_periods:
        raise ValueError("insufficient history for configured blind holdout")
    holdout_periods = min(desired, max_holdout)
    holdout_start = dates[-holdout_periods]
    timestamps = pd.to_datetime(bars["timestamp"], utc=True)
    research = bars.loc[timestamps < holdout_start].copy()
    return research, pd.Timestamp(holdout_start)


def evaluate_blind_holdout(
    bars: pd.DataFrame,
    model_config: ModelConfig,
    *,
    horizon: int,
    holdout_start: pd.Timestamp,
    config: HoldoutConfig | None = None,
    objective: ObjectiveWeights | None = None,
    top_fraction: float = 0.30,
    weighting: str = "equal",
) -> HoldoutReport:
    """Fit only on pre-holdout observations and score only the untouched final block."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    cfg = config or HoldoutConfig()
    panel = build_panel(bars)
    features = add_cross_sectional_features(panel).loc[:, FEATURE_COLUMNS]
    labels = make_panel_labels(panel, horizons=(horizon,))[f"fwd_return_{horizon}"]
    labels.name = f"fwd_return_{horizon}"

    row_dates = pd.DatetimeIndex(features.index.get_level_values("timestamp"))
    unique_dates = pd.DatetimeIndex(row_dates.unique()).sort_values()
    holdout_start = pd.Timestamp(holdout_start)
    if holdout_start.tzinfo is None:
        holdout_start = holdout_start.tz_localize("UTC")
    else:
        holdout_start = holdout_start.tz_convert("UTC")

    research_dates = unique_dates[unique_dates < holdout_start]
    holdout_dates = unique_dates[unique_dates >= holdout_start]
    if len(holdout_dates) < cfg.min_holdout_periods:
        raise ValueError("blind holdout is shorter than configured minimum")
    if len(research_dates) <= horizon:
        raise ValueError("insufficient pre-holdout history after label purge")

    train_dates = research_dates[:-horizon]
    train_mask = row_dates.isin(train_dates)
    holdout_mask = row_dates.isin(holdout_dates)

    X_train = features.loc[train_mask]
    y_train = labels.loc[train_mask]
    X_test = features.loc[holdout_mask]
    valid_train = X_train.notna().all(axis=1) & y_train.notna() & np.isfinite(y_train.astype(float))
    valid_test = X_test.notna().all(axis=1)
    if int(valid_train.sum()) < 20 or int(valid_test.sum()) == 0:
        raise ValueError("insufficient valid rows for blind holdout evaluation")

    model = build_model(model_config)
    model.fit(
        X_train.loc[valid_train].to_numpy(dtype=float),
        y_train.loc[valid_train].to_numpy(dtype=float),
    )
    predicted = np.asarray(
        model.predict(X_test.loc[valid_test].to_numpy(dtype=float)),
        dtype=float,
    )
    if not np.all(np.isfinite(predicted)):
        raise ValueError("model produced non-finite blind-holdout predictions")

    predictions = pd.Series(np.nan, index=features.index, dtype=float)
    predictions.loc[X_test.loc[valid_test].index] = predicted
    metrics, robustness, net_returns, turnover = _evaluate_panel_predictions(
        panel,
        predictions,
        ExperimentConfig(
            model_config,
            top_fraction=top_fraction,
            weighting=weighting,
        ),
    )
    coverage = float(predictions.loc[holdout_mask].notna().mean())
    stress = evaluate_stress_suite(net_returns, turnover=turnover)
    score = risk_adjusted_objective(
        metrics,
        robustness=robustness,
        oos_coverage=coverage,
        stress_score=stress.score,
        weights=objective,
    )

    reasons: list[str] = []
    if coverage < cfg.min_prediction_coverage:
        reasons.append("holdout_prediction_coverage")
    if float(metrics.get("max_drawdown", 1.0)) > cfg.max_drawdown:
        reasons.append("holdout_drawdown")
    if stress.score < cfg.min_stress_score:
        reasons.append("holdout_stress_failure")
    if score <= cfg.min_score:
        reasons.append("holdout_non_positive_score")

    return HoldoutReport(
        holdout_start=holdout_start,
        periods=len(holdout_dates),
        metrics={key: float(value) for key, value in metrics.items()},
        robustness=float(robustness),
        prediction_coverage=coverage,
        stress_score=float(stress.score),
        stress_report=stress,
        score=float(score),
        passed=not reasons,
        reasons=tuple(reasons),
    )
