from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockbot.arena.experiments import ExperimentConfig, _evaluate_panel_predictions
from stockbot.data.panel import build_panel
from stockbot.data.schemas import DatasetMetadata
from stockbot.domain.models import MarketRegime
from stockbot.features.cross_sectional import add_cross_sectional_features
from stockbot.ml.labels import make_panel_labels
from stockbot.ml.models import ModelConfig, build_model
from stockbot.ml.purged_cv import PurgedWalkForwardSplitter
from stockbot.research.objective import risk_adjusted_objective
from stockbot.research.regime_eval import build_market_regime_series
from stockbot.research.stress import StressReport, evaluate_stress_suite
from stockbot.research.training_pipeline import FEATURE_COLUMNS


@dataclass(frozen=True)
class RegimeSpecialistResult:
    regime: MarketRegime
    horizon: int
    model_config: ModelConfig
    metrics: dict[str, float]
    robustness: float
    oos_coverage: float
    stress_report: StressReport
    score: float
    net_returns: pd.Series
    turnover_series: pd.Series


def run_regime_specialist(
    bars: pd.DataFrame,
    metadata: DatasetMetadata,
    model_config: ModelConfig,
    *,
    regime: MarketRegime,
    horizon: int,
) -> RegimeSpecialistResult:
    """Train and evaluate a model only on observations belonging to one market regime.

    The specialist uses the same causal features and purged walk-forward structure as
    the general research pipeline. Training rows and OOS prediction rows are both
    restricted to the target regime, so each specialist is judged only where it is
    intended to be active.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    panel = build_panel(bars)
    features = add_cross_sectional_features(panel).loc[:, FEATURE_COLUMNS]
    labels = make_panel_labels(panel, horizons=(horizon,))[f"fwd_return_{horizon}"]
    labels.name = f"fwd_return_{horizon}"

    regime_daily = build_market_regime_series(bars)
    row_dates = pd.DatetimeIndex(features.index.get_level_values("timestamp"))
    row_regimes = pd.Series(row_dates.map(regime_daily), index=features.index, dtype="object")
    target_value = regime.value

    n_dates = len(panel.index.get_level_values("timestamp").unique())
    train_periods = max(60, min(126, n_dates // 2))
    test_periods = max(10, min(21, n_dates // 10))
    splitter = PurgedWalkForwardSplitter(train_periods, test_periods, horizon, horizon)

    predictions = pd.Series(np.nan, index=features.index, dtype=float, name=f"pred_{target_value}")
    target_rows = row_regimes.eq(target_value)
    eligible_target_rows = int(target_rows.sum())

    for train_idx, test_idx in splitter.split(features.index):
        X_train = features.iloc[train_idx]
        y_train = labels.iloc[train_idx]
        X_test = features.iloc[test_idx]
        train_regime = row_regimes.iloc[train_idx].eq(target_value)
        test_regime = row_regimes.iloc[test_idx].eq(target_value)
        valid_train = (
            X_train.notna().all(axis=1)
            & y_train.notna()
            & np.isfinite(y_train.astype(float))
            & train_regime
        )
        valid_test = X_test.notna().all(axis=1) & test_regime
        if int(valid_train.sum()) < 20 or int(valid_test.sum()) == 0:
            continue

        model = build_model(model_config)
        model.fit(
            X_train.loc[valid_train].to_numpy(dtype=float),
            y_train.loc[valid_train].to_numpy(dtype=float),
        )
        pred = np.asarray(model.predict(X_test.loc[valid_test].to_numpy(dtype=float)), dtype=float)
        if not np.all(np.isfinite(pred)):
            raise ValueError("regime specialist produced non-finite OOS predictions")
        predictions.loc[X_test.loc[valid_test].index] = pred

    metrics, robustness, net_returns, turnover = _evaluate_panel_predictions(
        panel,
        predictions,
        ExperimentConfig(model_config),
    )
    predicted_target_rows = int(predictions.loc[target_rows].notna().sum()) if eligible_target_rows else 0
    coverage = predicted_target_rows / eligible_target_rows if eligible_target_rows else 0.0
    stress = evaluate_stress_suite(net_returns, turnover=turnover)
    score = risk_adjusted_objective(
        metrics,
        robustness=robustness,
        oos_coverage=coverage,
        stress_score=stress.score,
    )

    return RegimeSpecialistResult(
        regime=regime,
        horizon=int(horizon),
        model_config=model_config,
        metrics={key: float(value) for key, value in metrics.items()},
        robustness=float(robustness),
        oos_coverage=float(coverage),
        stress_report=stress,
        score=float(score),
        net_returns=net_returns,
        turnover_series=turnover,
    )


def select_best_regime_specialists(
    bars: pd.DataFrame,
    metadata: DatasetMetadata,
    candidates: list[tuple[int, ModelConfig]],
) -> dict[str, RegimeSpecialistResult]:
    """Evaluate candidate model/horizon pairs as specialists and keep one per regime."""

    best: dict[str, RegimeSpecialistResult] = {}
    for regime in MarketRegime:
        for horizon, model_config in candidates:
            result = run_regime_specialist(
                bars,
                metadata,
                model_config,
                regime=regime,
                horizon=horizon,
            )
            incumbent = best.get(regime.value)
            if incumbent is None or result.score > incumbent.score:
                best[regime.value] = result
    return best
