import numpy as np
import pandas as pd

from stockbot.data.panel import build_panel
from stockbot.data.universe import PointInTimeUniverse, UniverseManifest, UniverseMembership
from stockbot.research.neutralization import evaluate_sector_factor_neutralization


def _bars_predictions_and_universe():
    rng = np.random.default_rng(20260908)
    dates = pd.date_range("2024-01-02", periods=180, freq="B", tz="UTC")
    symbols = ("AAA", "AAB", "BBB", "BBC", "CCC", "CCD")
    sectors = {
        "AAA": "Technology",
        "AAB": "Technology",
        "BBB": "Industrials",
        "BBC": "Industrials",
        "CCC": "Financials",
        "CCD": "Financials",
    }
    sector_bias = {"Technology": 0.020, "Industrials": -0.015, "Financials": 0.005}
    rows = []
    prediction_map = {}
    market = rng.normal(0.0002, 0.004, len(dates))
    for j, symbol in enumerate(symbols):
        local = market + rng.normal(0.0, 0.004, len(dates))
        close = 100.0 * np.exp(np.cumsum(local))
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": float(close[i]),
                    "high": float(close[i] * 1.006),
                    "low": float(close[i] * 0.994),
                    "close": float(close[i]),
                    "volume": int(700_000 + 40_000 * j + 2_000 * (i % 19)),
                }
            )
            prediction_map[(dt, symbol)] = float(
                sector_bias[sectors[symbol]]
                + 0.010 * np.sin(i / 11.0 + j)
                + rng.normal(0.0, 0.001)
            )

    predictions = pd.Series(
        prediction_map,
        index=pd.MultiIndex.from_tuples(prediction_map.keys(), names=["timestamp", "symbol"]),
        dtype=float,
    )
    memberships = tuple(
        UniverseMembership(
            symbol=symbol,
            effective_from="2023-01-01",
            sector=sectors[symbol],
            industry=f"{sectors[symbol]} Industry",
        )
        for symbol in symbols
    )
    universe = PointInTimeUniverse(
        memberships,
        UniverseManifest(
            source="synthetic",
            survivorship_bias_controlled=True,
            includes_delisted_securities=True,
            point_in_time_membership=True,
        ),
    )
    return pd.DataFrame(rows), predictions, universe


def test_sector_factor_neutralization_reduces_sector_bias_and_keeps_oos_signal():
    bars, predictions, universe = _bars_predictions_and_universe()
    report = evaluate_sector_factor_neutralization(bars, predictions, universe)

    assert report.sector_coverage == 1.0
    assert report.prediction_coverage > 0.50
    assert report.average_abs_sector_mean_after < report.average_abs_sector_mean_before
    assert np.isfinite(report.baseline_score)
    assert np.isfinite(report.neutralized_score)
    assert np.isfinite(report.neutralized_metrics["sharpe"])
    assert report.neutralized_predictions.notna().sum() > 0
    for name, before in report.factor_correlations_before.items():
        after = report.factor_correlations_after[name]
        assert abs(after) <= abs(before) + 0.10


def test_sector_factor_neutralization_fails_closed_when_sector_history_missing():
    bars, predictions, universe = _bars_predictions_and_universe()
    incomplete = PointInTimeUniverse(
        tuple(row for row in universe.memberships if row.symbol not in {"CCC", "CCD"}),
        universe.manifest,
    )
    try:
        evaluate_sector_factor_neutralization(bars, predictions, incomplete)
    except ValueError as exc:
        assert "sector coverage" in str(exc)
    else:
        raise AssertionError("expected insufficient sector coverage to fail closed")


def test_prediction_index_contract_accepts_reversed_multiindex():
    bars, predictions, universe = _bars_predictions_and_universe()
    reversed_predictions = predictions.copy()
    reversed_predictions.index = reversed_predictions.index.reorder_levels([1, 0])
    report = evaluate_sector_factor_neutralization(bars, reversed_predictions, universe)
    assert report.neutralized_predictions.index.equals(build_panel(bars).index)
