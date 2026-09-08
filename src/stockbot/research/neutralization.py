from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockbot.arena.experiments import ExperimentConfig, _evaluate_panel_predictions
from stockbot.data.panel import build_panel
from stockbot.data.universe import PointInTimeUniverse
from stockbot.features.cross_sectional import add_cross_sectional_features
from stockbot.ml.models import ModelConfig
from stockbot.research.objective import risk_adjusted_objective
from stockbot.research.stress import StressReport, evaluate_stress_suite


@dataclass(frozen=True)
class NeutralizationConfig:
    factor_columns: tuple[str, ...] = (
        "momentum_20",
        "realized_vol_20",
        "log_dollar_volume",
    )
    min_sector_coverage: float = 0.90
    min_cross_section: int = 4
    ridge: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 < self.min_sector_coverage <= 1.0:
            raise ValueError("min_sector_coverage must be in (0,1]")
        if self.min_cross_section < 2:
            raise ValueError("min_cross_section must be at least 2")
        if self.ridge < 0.0:
            raise ValueError("ridge cannot be negative")
        if not self.factor_columns:
            raise ValueError("factor_columns cannot be empty")


@dataclass(frozen=True)
class NeutralizationReport:
    baseline_metrics: dict[str, float]
    neutralized_metrics: dict[str, float]
    baseline_score: float
    neutralized_score: float
    score_delta: float
    baseline_stress: StressReport
    neutralized_stress: StressReport
    sector_coverage: float
    prediction_coverage: float
    average_abs_sector_mean_before: float
    average_abs_sector_mean_after: float
    factor_correlations_before: dict[str, float]
    factor_correlations_after: dict[str, float]
    neutralized_predictions: pd.Series
    baseline_net_returns: pd.Series
    neutralized_net_returns: pd.Series


def _normalize_prediction_index(predictions: pd.Series, panel_index: pd.MultiIndex) -> pd.Series:
    series = pd.Series(predictions, dtype=float).copy()
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
        raise ValueError("predictions must use a two-level MultiIndex")
    names = tuple(series.index.names)
    if names == ("symbol", "timestamp"):
        series.index = series.index.reorder_levels([1, 0])
    elif names != ("timestamp", "symbol"):
        raise ValueError("prediction index levels must be timestamp/symbol or symbol/timestamp")
    timestamps = pd.to_datetime(series.index.get_level_values("timestamp"), utc=True)
    symbols = series.index.get_level_values("symbol").astype(str).str.upper()
    series.index = pd.MultiIndex.from_arrays([timestamps, symbols], names=["timestamp", "symbol"])
    if series.index.duplicated().any():
        raise ValueError("predictions contain duplicate timestamp/symbol rows")
    return series.sort_index().reindex(panel_index)


def _sector_series(index: pd.MultiIndex, universe: PointInTimeUniverse) -> pd.Series:
    sectors: list[str | None] = []
    for timestamp, symbol in index:
        metadata = universe.metadata_at(str(symbol), pd.Timestamp(timestamp))
        sector = None if metadata is None else metadata.sector
        sectors.append(None if sector is None or not str(sector).strip() else str(sector).strip())
    return pd.Series(sectors, index=index, dtype="object", name="sector")


def _standardize(frame: pd.DataFrame) -> pd.DataFrame:
    means = frame.mean(axis=0)
    std = frame.std(axis=0, ddof=0).replace(0.0, np.nan)
    return frame.sub(means, axis=1).div(std, axis=1).fillna(0.0)


def _residualize_date(
    prediction: pd.Series,
    factors: pd.DataFrame,
    sectors: pd.Series,
    *,
    ridge: float,
) -> pd.Series:
    sector_dummies = pd.get_dummies(sectors.astype(str), prefix="sector", dtype=float)
    if sector_dummies.shape[1] > 1:
        sector_dummies = sector_dummies.iloc[:, 1:]
    else:
        sector_dummies = sector_dummies.iloc[:, 0:0]
    factor_values = _standardize(factors.astype(float))
    x_frame = pd.concat([sector_dummies, factor_values], axis=1)
    x = np.column_stack([np.ones(len(prediction)), x_frame.to_numpy(dtype=float)])
    y = prediction.to_numpy(dtype=float)
    penalty = np.eye(x.shape[1], dtype=float) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y
    residual = y - x @ coefficients
    return pd.Series(residual, index=prediction.index, dtype=float)


def _sector_mean_magnitude(predictions: pd.Series, sectors: pd.Series) -> float:
    frame = pd.concat([predictions.rename("prediction"), sectors], axis=1).dropna()
    if frame.empty:
        return 0.0
    grouped = frame.groupby([frame.index.get_level_values("timestamp"), "sector"])["prediction"].mean()
    return float(grouped.abs().mean()) if len(grouped) else 0.0


def _factor_correlations(predictions: pd.Series, factors: pd.DataFrame) -> dict[str, float]:
    output: dict[str, float] = {}
    for column in factors.columns:
        aligned = pd.concat([predictions.rename("prediction"), factors[column]], axis=1).dropna()
        if len(aligned) < 3 or aligned["prediction"].std(ddof=0) == 0.0 or aligned[column].std(ddof=0) == 0.0:
            output[column] = 0.0
        else:
            value = aligned["prediction"].corr(aligned[column])
            output[column] = 0.0 if not np.isfinite(value) else float(value)
    return output


def evaluate_sector_factor_neutralization(
    bars: pd.DataFrame,
    predictions: pd.Series,
    universe: PointInTimeUniverse,
    *,
    config: NeutralizationConfig | None = None,
    top_fraction: float = 0.30,
    weighting: str = "conviction",
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
) -> NeutralizationReport:
    """Compare an OOS signal with a point-in-time sector/factor residualized version.

    Neutralization is performed independently inside each timestamp. Sector labels are
    queried as-of that timestamp and factor columns come from StockBot's causal feature
    pipeline. This routine is diagnostic only and does not promote a strategy.
    """

    cfg = config or NeutralizationConfig()
    panel = build_panel(bars)
    normalized = _normalize_prediction_index(predictions, panel.index)
    all_features = add_cross_sectional_features(panel)
    missing = sorted(set(cfg.factor_columns).difference(all_features.columns))
    if missing:
        raise ValueError(f"unknown neutralization factor columns: {missing}")
    factors = all_features.loc[:, cfg.factor_columns].replace([np.inf, -np.inf], np.nan)
    sectors = _sector_series(panel.index, universe)

    predicted_mask = normalized.notna() & np.isfinite(normalized)
    predicted_count = int(predicted_mask.sum())
    if predicted_count == 0:
        raise ValueError("neutralization requires retained OOS predictions")
    sector_coverage = float(sectors.loc[predicted_mask].notna().mean())
    if sector_coverage < cfg.min_sector_coverage:
        raise ValueError(
            f"point-in-time sector coverage {sector_coverage:.3f} is below required {cfg.min_sector_coverage:.3f}"
        )

    residualized = pd.Series(np.nan, index=panel.index, dtype=float, name="neutralized_prediction")
    timestamps = panel.index.get_level_values("timestamp").unique()
    for timestamp in timestamps:
        date_index = panel.loc[(timestamp, slice(None)), :].index
        valid = (
            normalized.loc[date_index].notna()
            & sectors.loc[date_index].notna()
            & factors.loc[date_index].notna().all(axis=1)
        )
        valid_index = date_index[valid.to_numpy()]
        if len(valid_index) < cfg.min_cross_section:
            continue
        residualized.loc[valid_index] = _residualize_date(
            normalized.loc[valid_index],
            factors.loc[valid_index],
            sectors.loc[valid_index],
            ridge=cfg.ridge,
        )

    neutralized_count = int(residualized.notna().sum())
    prediction_coverage = neutralized_count / predicted_count if predicted_count else 0.0
    if neutralized_count == 0:
        raise ValueError("neutralization produced no valid cross-sectional predictions")

    experiment_config = ExperimentConfig(
        ModelConfig("ridge", {"alpha": 1.0}, seed=7),
        top_fraction=top_fraction,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        weighting=weighting,
    )
    baseline_metrics, baseline_robustness, baseline_returns, baseline_turnover = _evaluate_panel_predictions(
        panel, normalized, experiment_config
    )
    neutral_metrics, neutral_robustness, neutral_returns, neutral_turnover = _evaluate_panel_predictions(
        panel, residualized, experiment_config
    )
    baseline_stress = evaluate_stress_suite(baseline_returns, turnover=baseline_turnover)
    neutral_stress = evaluate_stress_suite(neutral_returns, turnover=neutral_turnover)
    baseline_oos = float(normalized.notna().mean())
    neutral_oos = float(residualized.notna().mean())
    baseline_score = risk_adjusted_objective(
        baseline_metrics,
        robustness=baseline_robustness,
        oos_coverage=baseline_oos,
        stress_score=baseline_stress.score,
    )
    neutral_score = risk_adjusted_objective(
        neutral_metrics,
        robustness=neutral_robustness,
        oos_coverage=neutral_oos,
        stress_score=neutral_stress.score,
    )

    return NeutralizationReport(
        baseline_metrics={key: float(value) for key, value in baseline_metrics.items()},
        neutralized_metrics={key: float(value) for key, value in neutral_metrics.items()},
        baseline_score=float(baseline_score),
        neutralized_score=float(neutral_score),
        score_delta=float(neutral_score - baseline_score),
        baseline_stress=baseline_stress,
        neutralized_stress=neutral_stress,
        sector_coverage=sector_coverage,
        prediction_coverage=float(prediction_coverage),
        average_abs_sector_mean_before=_sector_mean_magnitude(normalized, sectors),
        average_abs_sector_mean_after=_sector_mean_magnitude(residualized, sectors),
        factor_correlations_before=_factor_correlations(normalized, factors),
        factor_correlations_after=_factor_correlations(residualized, factors),
        neutralized_predictions=residualized,
        baseline_net_returns=baseline_returns,
        neutralized_net_returns=neutral_returns,
    )
