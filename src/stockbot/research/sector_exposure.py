from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockbot.arena.experiments import _signal_weights
from stockbot.data.panel import build_panel
from stockbot.data.universe import PointInTimeUniverse


@dataclass(frozen=True)
class SectorPortfolioExposureReport:
    """Point-in-time sector exposure of StockBot's lagged executed portfolio."""

    sector_coverage: float
    active_dates: int
    average_max_sector_weight: float
    worst_max_sector_weight: float
    average_sector_hhi: float
    average_active_sectors: float
    average_unclassified_weight: float
    average_sector_weights: dict[str, float]
    daily_sector_weights: pd.DataFrame
    executed_weights: pd.DataFrame


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
    series.index = pd.MultiIndex.from_arrays(
        [timestamps, symbols],
        names=["timestamp", "symbol"],
    )
    if series.index.duplicated().any():
        raise ValueError("predictions contain duplicate timestamp/symbol rows")
    normalized = series.sort_index().reindex(panel_index)
    if int(np.isfinite(normalized.to_numpy(dtype=float, na_value=np.nan)).sum()) == 0:
        raise ValueError("sector exposure requires retained OOS predictions")
    return normalized


def evaluate_sector_portfolio_exposure(
    bars: pd.DataFrame,
    predictions: pd.Series,
    universe: PointInTimeUniverse,
    *,
    top_fraction: float = 0.30,
    weighting: str = "conviction",
    min_sector_coverage: float = 0.90,
) -> SectorPortfolioExposureReport:
    """Audit sector concentration of the actually executed long-only portfolio.

    Portfolio weights are built with StockBot's canonical signal allocator and shifted
    by one bar before exposure is measured, matching ``_evaluate_panel_predictions``.
    Every holding is classified using universe metadata effective on the execution
    timestamp, so historical sector changes cannot leak backwards into the audit.
    """

    if not 0.0 < float(top_fraction) <= 1.0:
        raise ValueError("top_fraction must be in (0,1]")
    if weighting not in {"equal", "conviction"}:
        raise ValueError("weighting must be 'equal' or 'conviction'")
    if not 0.0 < float(min_sector_coverage) <= 1.0:
        raise ValueError("min_sector_coverage must be in (0,1]")

    panel = build_panel(bars)
    normalized = _normalize_prediction_index(predictions, panel.index)
    dates = panel.index.get_level_values("timestamp").unique().sort_values()
    symbols = panel.index.get_level_values("symbol").unique().sort_values()

    signal_weights = _signal_weights(
        normalized,
        float(top_fraction),
        weighting,
    ).reindex(index=dates, columns=symbols, fill_value=0.0).fillna(0.0)
    executed = signal_weights.shift(1).fillna(0.0).astype(float)

    gross_by_date = executed.abs().sum(axis=1)
    active_mask = gross_by_date > 0.0
    active_dates = executed.index[active_mask]
    if len(active_dates) == 0:
        raise ValueError("sector exposure produced no active executed portfolio dates")

    daily_rows: dict[pd.Timestamp, dict[str, float]] = {}
    unclassified_by_date: dict[pd.Timestamp, float] = {}
    total_gross = 0.0
    total_classified = 0.0

    for timestamp in active_dates:
        row = executed.loc[timestamp]
        gross = float(row.abs().sum())
        if gross <= 0.0 or not np.isfinite(gross):
            continue
        total_gross += gross
        sector_weights: dict[str, float] = {}
        unclassified = 0.0

        for symbol, raw_weight in row.items():
            weight = abs(float(raw_weight))
            if weight <= 0.0 or not np.isfinite(weight):
                continue
            metadata = universe.metadata_at(str(symbol), pd.Timestamp(timestamp))
            sector = None if metadata is None else metadata.sector
            if sector is None or not str(sector).strip():
                unclassified += weight
                continue
            name = str(sector).strip()
            sector_weights[name] = sector_weights.get(name, 0.0) + weight
            total_classified += weight

        daily_rows[pd.Timestamp(timestamp)] = {
            sector: weight / gross for sector, weight in sector_weights.items()
        }
        unclassified_by_date[pd.Timestamp(timestamp)] = unclassified / gross

    if total_gross <= 0.0:
        raise ValueError("sector exposure produced no measurable executed gross exposure")
    sector_coverage = total_classified / total_gross
    if sector_coverage < float(min_sector_coverage):
        raise ValueError(
            f"executed sector coverage {sector_coverage:.3f} is below required {float(min_sector_coverage):.3f}"
        )

    daily_sector_weights = pd.DataFrame.from_dict(daily_rows, orient="index").fillna(0.0)
    daily_sector_weights.index = pd.DatetimeIndex(
        pd.to_datetime(daily_sector_weights.index, utc=True),
        name="timestamp",
    )
    daily_sector_weights = daily_sector_weights.sort_index().sort_index(axis=1)

    if daily_sector_weights.shape[1] == 0:
        max_sector = pd.Series(0.0, index=daily_sector_weights.index, dtype=float)
        sector_hhi = pd.Series(0.0, index=daily_sector_weights.index, dtype=float)
        active_sectors = pd.Series(0.0, index=daily_sector_weights.index, dtype=float)
        average_sector_weights: dict[str, float] = {}
    else:
        max_sector = daily_sector_weights.max(axis=1)
        sector_hhi = daily_sector_weights.pow(2).sum(axis=1)
        active_sectors = daily_sector_weights.gt(0.0).sum(axis=1).astype(float)
        average_sector_weights = {
            str(sector): float(value)
            for sector, value in daily_sector_weights.mean(axis=0).items()
        }

    unclassified = pd.Series(unclassified_by_date, dtype=float).reindex(
        daily_sector_weights.index
    ).fillna(0.0)

    return SectorPortfolioExposureReport(
        sector_coverage=float(sector_coverage),
        active_dates=int(len(daily_sector_weights)),
        average_max_sector_weight=float(max_sector.mean()),
        worst_max_sector_weight=float(max_sector.max()),
        average_sector_hhi=float(sector_hhi.mean()),
        average_active_sectors=float(active_sectors.mean()),
        average_unclassified_weight=float(unclassified.mean()),
        average_sector_weights=average_sector_weights,
        daily_sector_weights=daily_sector_weights,
        executed_weights=executed.copy(),
    )
