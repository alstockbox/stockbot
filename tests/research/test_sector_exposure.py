import numpy as np
import pandas as pd
import pytest

from stockbot.data.universe import PointInTimeUniverse, UniverseManifest, UniverseMembership
from stockbot.research.sector_exposure import evaluate_sector_portfolio_exposure


def _bars(symbols=("AAA", "BBB"), periods=8):
    rows = []
    dates = pd.date_range("2025-01-02", periods=periods, freq="B", tz="UTC")
    for j, symbol in enumerate(symbols):
        for i, dt in enumerate(dates):
            close = 100.0 + i + j
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows), dates


def _predictions(dates, values):
    mapping = {}
    for dt in dates:
        for symbol, value in values.items():
            mapping[(dt, symbol)] = float(value)
    return pd.Series(
        mapping,
        index=pd.MultiIndex.from_tuples(mapping.keys(), names=["timestamp", "symbol"]),
        dtype=float,
    )


def _manifest():
    return UniverseManifest(
        source="synthetic",
        survivorship_bias_controlled=True,
        includes_delisted_securities=True,
        point_in_time_membership=True,
    )


def test_sector_exposure_uses_classification_as_of_execution_date():
    bars, dates = _bars()
    predictions = _predictions(dates, {"AAA": 1.0, "BBB": -1.0})
    universe = PointInTimeUniverse(
        (
            UniverseMembership("AAA", "2020-01-01", "2025-01-06", sector="Technology"),
            UniverseMembership("AAA", "2025-01-06", sector="Financials"),
            UniverseMembership("BBB", "2020-01-01", sector="Industrials"),
        ),
        _manifest(),
    )

    report = evaluate_sector_portfolio_exposure(
        bars,
        predictions,
        universe,
        top_fraction=0.50,
        weighting="equal",
    )

    # Signal weights are executed with the same one-bar lag used by StockBot's arena.
    assert report.daily_sector_weights.loc[pd.Timestamp("2025-01-03", tz="UTC"), "Technology"] == pytest.approx(1.0)
    assert report.daily_sector_weights.loc[pd.Timestamp("2025-01-06", tz="UTC"), "Financials"] == pytest.approx(1.0)
    assert report.daily_sector_weights.loc[pd.Timestamp("2025-01-06", tz="UTC")].get("Technology", 0.0) == pytest.approx(0.0)
    assert report.sector_coverage == pytest.approx(1.0)
    assert report.average_max_sector_weight == pytest.approx(1.0)
    assert report.worst_max_sector_weight == pytest.approx(1.0)


def test_sector_exposure_reports_actual_portfolio_concentration():
    symbols = ("AAA", "AAB", "BBB", "BBC")
    bars, dates = _bars(symbols=symbols)
    predictions = _predictions(dates, {symbol: 1.0 for symbol in symbols})
    universe = PointInTimeUniverse(
        (
            UniverseMembership("AAA", "2020-01-01", sector="Technology"),
            UniverseMembership("AAB", "2020-01-01", sector="Technology"),
            UniverseMembership("BBB", "2020-01-01", sector="Industrials"),
            UniverseMembership("BBC", "2020-01-01", sector="Industrials"),
        ),
        _manifest(),
    )

    report = evaluate_sector_portfolio_exposure(
        bars,
        predictions,
        universe,
        top_fraction=1.0,
        weighting="equal",
    )

    assert report.active_dates == len(dates) - 1
    assert report.average_max_sector_weight == pytest.approx(0.50)
    assert report.worst_max_sector_weight == pytest.approx(0.50)
    assert report.average_sector_hhi == pytest.approx(0.50)
    assert report.average_active_sectors == pytest.approx(2.0)
    assert report.average_unclassified_weight == pytest.approx(0.0)
    assert report.average_sector_weights["Technology"] == pytest.approx(0.50)
    assert report.average_sector_weights["Industrials"] == pytest.approx(0.50)


def test_sector_exposure_fails_closed_when_executed_holdings_lack_sector_history():
    bars, dates = _bars()
    predictions = _predictions(dates, {"AAA": 1.0, "BBB": 1.0})
    universe = PointInTimeUniverse(
        (UniverseMembership("AAA", "2020-01-01", sector="Technology"),),
        _manifest(),
    )

    with pytest.raises(ValueError, match="executed sector coverage"):
        evaluate_sector_portfolio_exposure(
            bars,
            predictions,
            universe,
            top_fraction=1.0,
            weighting="equal",
            min_sector_coverage=0.90,
        )
