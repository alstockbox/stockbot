from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockbot.data.point_in_time_features import PointInTimeFeatureStore
from stockbot.data.schemas import DatasetMetadata
from stockbot.ml.models import ModelConfig
from stockbot.research.training_pipeline import run_training_research


@dataclass(frozen=True)
class WindowResult:
    train_periods: int
    test_periods: int
    score: float
    robustness: float
    oos_coverage: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class WindowRobustnessReport:
    results: tuple[WindowResult, ...]
    score: float
    score_std: float
    worst_score: float
    positive_fraction: float


def evaluate_training_window_robustness(
    bars: pd.DataFrame,
    metadata: DatasetMetadata,
    model_config: ModelConfig,
    *,
    horizon: int,
    train_windows: tuple[int, ...] = (126, 252, 504),
    test_periods: int = 21,
    feature_columns: tuple[str, ...] | list[str] | None = None,
    auxiliary_store: PointInTimeFeatureStore | None = None,
    auxiliary_feature_names: tuple[str, ...] | list[str] | None = None,
    auxiliary_max_age_days: int | None = None,
    auxiliary_min_coverage: float = 0.80,
) -> WindowRobustnessReport:
    """Re-run one challenger across multiple walk-forward memory lengths."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not train_windows or any(window <= 0 for window in train_windows):
        raise ValueError("train_windows must contain positive values")
    if test_periods <= 0:
        raise ValueError("test_periods must be positive")

    timestamps = pd.DatetimeIndex(pd.to_datetime(bars["timestamp"], utc=True).unique()).sort_values()
    n_dates = len(timestamps)
    results: list[WindowResult] = []

    for window in sorted(set(train_windows)):
        if window + horizon + test_periods > n_dates:
            continue
        run = run_training_research(
            bars,
            metadata,
            model_configs=(model_config,),
            horizon=horizon,
            max_workers=1,
            train_periods=window,
            test_periods=test_periods,
            feature_columns=feature_columns,
            auxiliary_store=auxiliary_store,
            auxiliary_feature_names=auxiliary_feature_names,
            auxiliary_max_age_days=auxiliary_max_age_days,
            auxiliary_min_coverage=auxiliary_min_coverage,
        )
        if not run.leaderboard:
            continue
        result = run.leaderboard[0]
        results.append(
            WindowResult(
                train_periods=int(window),
                test_periods=int(test_periods),
                score=float(result.score),
                robustness=float(result.robustness),
                oos_coverage=float(result.oos_coverage),
                metrics={key: float(value) for key, value in result.metrics.items()},
            )
        )

    if not results:
        return WindowRobustnessReport((), 0.0, float("inf"), float("-inf"), 0.0)

    scores = np.asarray([result.score for result in results], dtype=float)
    normalized = np.clip((scores + 2.0) / 6.0, 0.0, 1.0)
    mean_component = float(normalized.mean())
    worst_component = float(normalized.min())
    stability_component = float(np.clip(1.0 - normalized.std(ddof=0), 0.0, 1.0))
    positive_fraction = float((scores > 0.0).mean())
    aggregate = float(
        np.clip(
            0.35 * mean_component
            + 0.30 * worst_component
            + 0.20 * stability_component
            + 0.15 * positive_fraction,
            0.0,
            1.0,
        )
    )
    return WindowRobustnessReport(
        results=tuple(results),
        score=aggregate,
        score_std=float(scores.std(ddof=0)),
        worst_score=float(scores.min()),
        positive_fraction=positive_fraction,
    )
