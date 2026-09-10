from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from stockbot.arena.experiments import ExperimentConfig, _evaluate_panel_predictions
from stockbot.data.panel import build_panel
from stockbot.ml.labels import make_panel_labels
from stockbot.ml.models import ModelConfig, build_model
from stockbot.ml.purged_cv import PurgedWalkForwardSplitter
from stockbot.research.objective import risk_adjusted_objective
from stockbot.research.stress import StressReport, evaluate_stress_suite


@dataclass(frozen=True)
class StackingReport:
    horizon: int
    member_ids: tuple[str, ...]
    meta_model: ModelConfig
    metrics: dict[str, float]
    robustness: float
    oos_coverage: float
    stress_report: StressReport
    score: float
    predictions: pd.Series
    net_returns: pd.Series
    turnover_series: pd.Series


def _normalize_prediction_index(predictions: pd.Series) -> pd.Series:
    series = pd.Series(predictions, dtype=float).copy()
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
        raise ValueError("stacking predictions require a two-level timestamp/symbol MultiIndex")
    names = tuple(series.index.names)
    if names == ("symbol", "timestamp"):
        series = series.reorder_levels(["timestamp", "symbol"])
    elif names != ("timestamp", "symbol"):
        raise ValueError("stacking prediction index levels must be named timestamp and symbol")
    timestamps = pd.to_datetime(series.index.get_level_values("timestamp"), utc=True)
    symbols = series.index.get_level_values("symbol").astype(str)
    series.index = pd.MultiIndex.from_arrays(
        [timestamps, symbols],
        names=["timestamp", "symbol"],
    )
    if series.index.has_duplicates:
        raise ValueError("stacking prediction index contains duplicate timestamp/symbol rows")
    return series.sort_index()


def _candidate_prediction_frame(candidates: list[Any] | tuple[Any, ...], horizon: int) -> pd.DataFrame:
    selected = [candidate for candidate in candidates if int(candidate.horizon) == int(horizon)]
    if len(selected) < 2:
        raise ValueError("stacking requires at least two candidates from the same horizon")

    series = []
    for candidate in selected:
        predictions = getattr(candidate.result, "predictions", None)
        if predictions is None or len(pd.Series(predictions).dropna()) == 0:
            raise ValueError("all stacking members require retained OOS predictions")
        normalized = _normalize_prediction_index(predictions)
        series.append(normalized.rename(str(candidate.experiment_id)))
    frame = pd.concat(series, axis=1, join="outer").sort_index()
    if frame.notna().all(axis=1).sum() == 0:
        raise ValueError("stacking members have no overlapping OOS prediction rows")
    return frame


def evaluate_oos_stacking(
    bars: pd.DataFrame,
    candidates: list[Any] | tuple[Any, ...],
    *,
    horizon: int,
    meta_model: ModelConfig | None = None,
    top_fraction: float = 0.30,
    weighting: str = "conviction",
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
) -> StackingReport:
    """Train a second-level model only on first-level OOS predictions.

    Base learners must already contain strictly OOS prediction streams. The meta-model
    receives those streams as features and is itself evaluated with a purged
    walk-forward split. No first-level in-sample predictions are created or used.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    member_candidates = [candidate for candidate in candidates if int(candidate.horizon) == int(horizon)]
    prediction_frame = _candidate_prediction_frame(member_candidates, horizon)
    member_ids = tuple(str(candidate.experiment_id) for candidate in member_candidates)

    panel = build_panel(bars)
    labels = make_panel_labels(panel, horizons=(horizon,))[f"fwd_return_{horizon}"]
    labels.name = f"fwd_return_{horizon}"
    prediction_frame = prediction_frame.reindex(panel.index)
    overlapping = int(prediction_frame.notna().all(axis=1).sum())
    if overlapping == 0:
        raise ValueError("stacking predictions do not overlap the market panel index")

    n_dates = len(panel.index.get_level_values("timestamp").unique())
    train_periods = max(60, min(252, n_dates // 2))
    test_periods = max(10, min(21, n_dates // 10))
    splitter = PurgedWalkForwardSplitter(train_periods, test_periods, horizon, horizon)
    model_config = meta_model or ModelConfig("ridge", {"alpha": 1.0}, seed=7)

    meta_predictions = pd.Series(np.nan, index=panel.index, dtype=float, name="stacked_prediction")
    eligible = prediction_frame.notna().all(axis=1) & labels.notna() & np.isfinite(labels.astype(float))

    for train_idx, test_idx in splitter.split(panel.index):
        X_train = prediction_frame.iloc[train_idx]
        y_train = labels.iloc[train_idx]
        X_test = prediction_frame.iloc[test_idx]
        valid_train = X_train.notna().all(axis=1) & y_train.notna() & np.isfinite(y_train.astype(float))
        valid_test = X_test.notna().all(axis=1)
        if int(valid_train.sum()) < 20 or int(valid_test.sum()) == 0:
            continue

        model = build_model(model_config)
        model.fit(
            X_train.loc[valid_train].to_numpy(dtype=float),
            y_train.loc[valid_train].to_numpy(dtype=float),
        )
        predictions = np.asarray(model.predict(X_test.loc[valid_test].to_numpy(dtype=float)), dtype=float)
        if not np.all(np.isfinite(predictions)):
            raise ValueError("stacking meta-model produced non-finite OOS predictions")
        meta_predictions.loc[X_test.loc[valid_test].index] = predictions

    predicted = int(meta_predictions.notna().sum())
    eligible_count = int(eligible.sum())
    if eligible_count > 0 and predicted == 0:
        raise ValueError("stacking produced no meta-level OOS predictions")
    oos_coverage = predicted / eligible_count if eligible_count else 0.0

    metrics, robustness, net_returns, turnover = _evaluate_panel_predictions(
        panel,
        meta_predictions,
        ExperimentConfig(
            model=model_config,
            top_fraction=top_fraction,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            weighting=weighting,
        ),
    )
    stress = evaluate_stress_suite(net_returns, turnover=turnover)
    score = risk_adjusted_objective(
        metrics,
        robustness=robustness,
        oos_coverage=oos_coverage,
        stress_score=stress.score,
    )

    return StackingReport(
        horizon=int(horizon),
        member_ids=member_ids,
        meta_model=model_config,
        metrics={key: float(value) for key, value in metrics.items()},
        robustness=float(robustness),
        oos_coverage=float(oos_coverage),
        stress_report=stress,
        score=float(score),
        predictions=meta_predictions,
        net_returns=net_returns,
        turnover_series=turnover,
    )
