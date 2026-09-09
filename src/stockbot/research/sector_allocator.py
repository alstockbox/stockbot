from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from stockbot.arena.experiments import ExperimentConfig, _evaluate_panel_predictions, _signal_weights
from stockbot.costs.model import LinearCostModel
from stockbot.data.panel import build_panel
from stockbot.data.universe import PointInTimeUniverse
from stockbot.evaluation.metrics import performance_metrics
from stockbot.ml.models import ModelConfig
from stockbot.research.objective import risk_adjusted_objective
from stockbot.research.sector_exposure import (
    SectorPortfolioExposureReport,
    evaluate_sector_portfolio_exposure,
)
from stockbot.research.stress import StressReport, evaluate_stress_suite


@dataclass(frozen=True)
class SectorCapAllocationReport:
    max_sector_weight: float
    baseline_metrics: dict[str, float]
    constrained_metrics: dict[str, float]
    baseline_score: float
    constrained_score: float
    score_delta: float
    baseline_stress: StressReport
    constrained_stress: StressReport
    baseline_sector_exposure: SectorPortfolioExposureReport
    constrained_sector_exposure: SectorPortfolioExposureReport
    average_signal_max_sector_weight_before: float
    average_signal_max_sector_weight_after: float
    average_cash_weight: float
    signal_diagnostics: pd.DataFrame
    constrained_signal_weights: pd.DataFrame
    constrained_executed_weights: pd.DataFrame
    constrained_net_returns: pd.Series
    constrained_turnover: pd.Series


def _cap_sector_totals(
    totals: dict[str, float],
    *,
    max_sector_weight: float,
    target_gross: float,
) -> dict[str, float]:
    target = {
        sector: min(float(weight), float(max_sector_weight))
        for sector, weight in totals.items()
    }
    remaining = max(0.0, float(target_gross) - sum(target.values()))
    tolerance = 1e-12
    for _ in range(max(1, len(target) * 3)):
        if remaining <= tolerance:
            break
        eligible = {
            sector: float(max_sector_weight) - current
            for sector, current in target.items()
            if float(max_sector_weight) - current > tolerance
        }
        if not eligible:
            break
        preference_total = sum(float(totals[sector]) for sector in eligible)
        if preference_total <= tolerance:
            preference = {
                sector: capacity / sum(eligible.values())
                for sector, capacity in eligible.items()
            }
        else:
            preference = {
                sector: float(totals[sector]) / preference_total
                for sector in eligible
            }

        allocated = 0.0
        for sector, capacity in eligible.items():
            addition = min(capacity, remaining * preference[sector])
            if addition > 0.0:
                target[sector] += addition
                allocated += addition
        if allocated <= tolerance:
            break
        remaining -= allocated
    return target


def apply_point_in_time_sector_cap(
    signal_weights: pd.DataFrame,
    universe: PointInTimeUniverse,
    *,
    max_sector_weight: float = 0.35,
    min_sector_coverage: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap long-only sector weights using only classifications known on signal date.

    Excess weight is redistributed across other active sectors up to their remaining
    capacity. If the active sector set cannot absorb all excess, the residual remains
    cash. No next-session classification is queried by this allocator.
    """

    if not 0.0 < float(max_sector_weight) <= 1.0:
        raise ValueError("max_sector_weight must be in (0,1]")
    if not 0.0 < float(min_sector_coverage) <= 1.0:
        raise ValueError("min_sector_coverage must be in (0,1]")
    if not isinstance(signal_weights.index, pd.DatetimeIndex):
        raise ValueError("signal_weights must use a DatetimeIndex")
    weights = signal_weights.copy().astype(float)
    weights.index = pd.to_datetime(weights.index, utc=True)
    weights.columns = [str(symbol).upper() for symbol in weights.columns]
    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("signal_weights must be finite")
    if (weights.to_numpy() < -1e-12).any():
        raise ValueError("sector cap currently supports long-only signal weights")
    weights = weights.clip(lower=0.0)

    constrained = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    diagnostics: dict[pd.Timestamp, dict[str, float]] = {}

    for timestamp, row in weights.iterrows():
        active = row[row > 0.0]
        gross = float(active.sum())
        if gross <= 0.0:
            diagnostics[pd.Timestamp(timestamp)] = {
                "sector_coverage": 1.0,
                "max_sector_weight_before": 0.0,
                "max_sector_weight_after": 0.0,
                "cash_weight_after": 1.0,
                "active_sectors_before": 0.0,
                "active_sectors_after": 0.0,
            }
            continue

        symbol_sector: dict[str, str] = {}
        classified_weight = 0.0
        sector_totals: dict[str, float] = {}
        for symbol, value in active.items():
            metadata = universe.metadata_at(str(symbol), pd.Timestamp(timestamp))
            sector = None if metadata is None else metadata.sector
            if sector is None or not str(sector).strip():
                continue
            sector_name = str(sector).strip()
            symbol_sector[str(symbol)] = sector_name
            classified_weight += float(value)
            sector_totals[sector_name] = sector_totals.get(sector_name, 0.0) + float(value)

        coverage = classified_weight / gross
        if coverage < float(min_sector_coverage):
            raise ValueError(
                f"signal-date sector coverage {coverage:.3f} is below required {float(min_sector_coverage):.3f}"
            )
        if classified_weight <= 0.0:
            raise ValueError("signal-date sector coverage produced no classified active weight")

        # Fail closed on the unclassified tail rather than silently allocating it.
        classified = active.loc[[symbol for symbol in active.index if str(symbol) in symbol_sector]]
        classified_gross = float(classified.sum())
        targets = _cap_sector_totals(
            sector_totals,
            max_sector_weight=float(max_sector_weight),
            target_gross=classified_gross,
        )

        for sector, target_weight in targets.items():
            members = [symbol for symbol in classified.index if symbol_sector[str(symbol)] == sector]
            original_total = float(classified.loc[members].sum())
            if original_total <= 0.0 or target_weight <= 0.0:
                continue
            constrained.loc[timestamp, members] = (
                classified.loc[members] * (target_weight / original_total)
            )

        after_sector_totals: dict[str, float] = {}
        for symbol, value in constrained.loc[timestamp].items():
            if float(value) <= 0.0:
                continue
            sector = symbol_sector.get(str(symbol))
            if sector is not None:
                after_sector_totals[sector] = after_sector_totals.get(sector, 0.0) + float(value)

        max_before = max(sector_totals.values(), default=0.0)
        max_after = max(after_sector_totals.values(), default=0.0)
        invested_after = float(constrained.loc[timestamp].sum())
        diagnostics[pd.Timestamp(timestamp)] = {
            "sector_coverage": float(coverage),
            "max_sector_weight_before": float(max_before),
            "max_sector_weight_after": float(max_after),
            "cash_weight_after": float(max(0.0, 1.0 - invested_after)),
            "active_sectors_before": float(sum(value > 0.0 for value in sector_totals.values())),
            "active_sectors_after": float(sum(value > 0.0 for value in after_sector_totals.values())),
        }

    diagnostic_frame = pd.DataFrame.from_dict(diagnostics, orient="index").sort_index()
    diagnostic_frame.index = pd.DatetimeIndex(diagnostic_frame.index, name="timestamp")
    return constrained, diagnostic_frame


def _audit_custom_signal_weights(
    signal_weights: pd.DataFrame,
    universe: PointInTimeUniverse,
    *,
    min_sector_coverage: float,
) -> SectorPortfolioExposureReport:
    executed = signal_weights.shift(1).fillna(0.0).astype(float)
    gross_by_date = executed.abs().sum(axis=1)
    active_dates = executed.index[gross_by_date > 0.0]
    if len(active_dates) == 0:
        raise ValueError("sector exposure produced no active executed portfolio dates")

    daily_rows: dict[pd.Timestamp, dict[str, float]] = {}
    unclassified_by_date: dict[pd.Timestamp, float] = {}
    total_gross = 0.0
    total_classified = 0.0
    for timestamp in active_dates:
        row = executed.loc[timestamp]
        gross = float(row.abs().sum())
        total_gross += gross
        sectors: dict[str, float] = {}
        unclassified = 0.0
        for symbol, raw_weight in row.items():
            weight = abs(float(raw_weight))
            if weight <= 0.0:
                continue
            metadata = universe.metadata_at(str(symbol), pd.Timestamp(timestamp))
            sector = None if metadata is None else metadata.sector
            if sector is None or not str(sector).strip():
                unclassified += weight
                continue
            name = str(sector).strip()
            sectors[name] = sectors.get(name, 0.0) + weight
            total_classified += weight
        daily_rows[pd.Timestamp(timestamp)] = {
            sector: weight / gross for sector, weight in sectors.items()
        }
        unclassified_by_date[pd.Timestamp(timestamp)] = unclassified / gross

    coverage = total_classified / total_gross if total_gross > 0.0 else 0.0
    if coverage < float(min_sector_coverage):
        raise ValueError(
            f"executed sector coverage {coverage:.3f} is below required {float(min_sector_coverage):.3f}"
        )

    daily = pd.DataFrame.from_dict(daily_rows, orient="index").fillna(0.0)
    daily.index = pd.DatetimeIndex(pd.to_datetime(daily.index, utc=True), name="timestamp")
    daily = daily.sort_index().sort_index(axis=1)
    max_sector = daily.max(axis=1) if daily.shape[1] else pd.Series(0.0, index=daily.index)
    hhi = daily.pow(2).sum(axis=1) if daily.shape[1] else pd.Series(0.0, index=daily.index)
    active_sector_count = daily.gt(0.0).sum(axis=1).astype(float) if daily.shape[1] else pd.Series(0.0, index=daily.index)
    unclassified = pd.Series(unclassified_by_date, dtype=float).reindex(daily.index).fillna(0.0)
    averages = {str(key): float(value) for key, value in daily.mean(axis=0).items()}

    return SectorPortfolioExposureReport(
        sector_coverage=float(coverage),
        active_dates=int(len(daily)),
        average_max_sector_weight=float(max_sector.mean()),
        worst_max_sector_weight=float(max_sector.max()),
        average_sector_hhi=float(hhi.mean()),
        average_active_sectors=float(active_sector_count.mean()),
        average_unclassified_weight=float(unclassified.mean()),
        average_sector_weights=averages,
        daily_sector_weights=daily,
        executed_weights=executed.copy(),
    )


def _evaluate_custom_signal_weights(
    panel: pd.DataFrame,
    signal_weights: pd.DataFrame,
    predictions: pd.Series,
    *,
    commission_bps: float,
    slippage_bps: float,
) -> tuple[dict[str, float], float, pd.Series, pd.Series, pd.DataFrame]:
    close = panel["close"].unstack("symbol").sort_index().astype(float)
    signals = signal_weights.reindex(index=close.index, columns=close.columns, fill_value=0.0).fillna(0.0)
    executed = signals.shift(1).fillna(0.0)
    asset_returns = close.pct_change().fillna(0.0)
    gross = (executed * asset_returns).sum(axis=1)
    turnover = executed.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = executed.iloc[0].abs().sum()

    cost_model = LinearCostModel(float(commission_bps), float(slippage_bps))
    cost_rate = turnover.apply(cost_model.rate_for_turnover)
    net = gross - cost_rate

    pred_dates = predictions.dropna().index.get_level_values("timestamp")
    if len(pred_dates):
        start = pd.Timestamp(pred_dates.min())
        net = net.loc[net.index >= start]
        turnover = turnover.loc[turnover.index >= start]
        executed = executed.loc[executed.index >= start]
        cost_rate = cost_rate.loc[cost_rate.index >= start]

    metrics = performance_metrics(net)
    metrics["turnover"] = float(turnover.sum())
    metrics["cost_ratio"] = float(cost_rate.sum())
    metrics["concentration"] = float(executed.pow(2).sum(axis=1).mean()) if len(executed) else 0.0
    monthly = (1.0 + net).resample("21B").prod() - 1.0 if len(net) else pd.Series(dtype=float)
    metrics["instability"] = max(
        0.0,
        min(1.0, float(monthly.std(ddof=0)) if len(monthly) > 1 else 0.0),
    )
    robustness = float(
        np.clip(
            (1.0 - metrics["max_drawdown"]) * (1.0 - metrics["instability"]),
            0.0,
            1.0,
        )
    )
    return metrics, robustness, net.copy(), turnover.copy(), executed.copy()


def evaluate_sector_cap_challenger(
    bars: pd.DataFrame,
    predictions: pd.Series,
    universe: PointInTimeUniverse,
    *,
    top_fraction: float = 0.30,
    weighting: str = "conviction",
    max_sector_weight: float = 0.35,
    min_sector_coverage: float = 0.90,
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
    oos_coverage: float = 1.0,
) -> SectorCapAllocationReport:
    """Compare canonical portfolio construction with a causal signal-date sector cap.

    This is a diagnostic challenger only. Sector metadata used for construction is
    queried at the signal timestamp. The resulting one-bar-lagged holdings are then
    audited independently using classifications effective on each execution timestamp.
    """

    if not 0.0 < float(oos_coverage) <= 1.0:
        raise ValueError("oos_coverage must be in (0,1]")
    panel = build_panel(bars)
    series = pd.Series(predictions, dtype=float).copy()
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
        raise ValueError("predictions must use a two-level MultiIndex")
    if tuple(series.index.names) == ("symbol", "timestamp"):
        series.index = series.index.reorder_levels([1, 0])
    elif tuple(series.index.names) != ("timestamp", "symbol"):
        raise ValueError("prediction index levels must be timestamp/symbol or symbol/timestamp")
    timestamps = pd.to_datetime(series.index.get_level_values("timestamp"), utc=True)
    symbols = series.index.get_level_values("symbol").astype(str).str.upper()
    series.index = pd.MultiIndex.from_arrays([timestamps, symbols], names=["timestamp", "symbol"])
    if series.index.duplicated().any():
        raise ValueError("predictions contain duplicate timestamp/symbol rows")
    series = series.sort_index().reindex(panel.index)
    if int(series.notna().sum()) == 0:
        raise ValueError("sector-cap challenger requires retained OOS predictions")

    baseline_config = ExperimentConfig(
        ModelConfig("ridge", {"alpha": 1.0}, seed=7),
        top_fraction=float(top_fraction),
        weighting=weighting,
        commission_bps=float(commission_bps),
        slippage_bps=float(slippage_bps),
    )
    baseline_metrics, baseline_robustness, baseline_returns, baseline_turnover = _evaluate_panel_predictions(
        panel,
        series,
        baseline_config,
    )
    baseline_stress = evaluate_stress_suite(baseline_returns, turnover=baseline_turnover)
    baseline_score = risk_adjusted_objective(
        baseline_metrics,
        robustness=baseline_robustness,
        oos_coverage=float(oos_coverage),
        stress_score=baseline_stress.score,
    )

    canonical_signal_weights = _signal_weights(series, float(top_fraction), weighting)
    constrained_signal_weights, diagnostics = apply_point_in_time_sector_cap(
        canonical_signal_weights,
        universe,
        max_sector_weight=float(max_sector_weight),
        min_sector_coverage=float(min_sector_coverage),
    )
    constrained_metrics, constrained_robustness, constrained_returns, constrained_turnover, executed = _evaluate_custom_signal_weights(
        panel,
        constrained_signal_weights,
        series,
        commission_bps=float(commission_bps),
        slippage_bps=float(slippage_bps),
    )
    constrained_stress = evaluate_stress_suite(constrained_returns, turnover=constrained_turnover)
    constrained_score = risk_adjusted_objective(
        constrained_metrics,
        robustness=constrained_robustness,
        oos_coverage=float(oos_coverage),
        stress_score=constrained_stress.score,
    )

    baseline_sector_exposure = evaluate_sector_portfolio_exposure(
        bars,
        series,
        universe,
        top_fraction=float(top_fraction),
        weighting=weighting,
        min_sector_coverage=float(min_sector_coverage),
    )
    constrained_sector_exposure = _audit_custom_signal_weights(
        constrained_signal_weights,
        universe,
        min_sector_coverage=float(min_sector_coverage),
    )

    active_diagnostics = diagnostics.loc[diagnostics["max_sector_weight_before"] > 0.0]
    before_average = float(active_diagnostics["max_sector_weight_before"].mean()) if len(active_diagnostics) else 0.0
    after_average = float(active_diagnostics["max_sector_weight_after"].mean()) if len(active_diagnostics) else 0.0
    cash_average = float(active_diagnostics["cash_weight_after"].mean()) if len(active_diagnostics) else 0.0

    return SectorCapAllocationReport(
        max_sector_weight=float(max_sector_weight),
        baseline_metrics={key: float(value) for key, value in baseline_metrics.items()},
        constrained_metrics={key: float(value) for key, value in constrained_metrics.items()},
        baseline_score=float(baseline_score),
        constrained_score=float(constrained_score),
        score_delta=float(constrained_score - baseline_score),
        baseline_stress=baseline_stress,
        constrained_stress=constrained_stress,
        baseline_sector_exposure=baseline_sector_exposure,
        constrained_sector_exposure=constrained_sector_exposure,
        average_signal_max_sector_weight_before=before_average,
        average_signal_max_sector_weight_after=after_average,
        average_cash_weight=cash_average,
        signal_diagnostics=diagnostics,
        constrained_signal_weights=constrained_signal_weights,
        constrained_executed_weights=executed,
        constrained_net_returns=constrained_returns,
        constrained_turnover=constrained_turnover,
    )
