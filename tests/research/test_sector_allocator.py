import numpy as np
import pandas as pd
import pytest

from stockbot.data.universe import PointInTimeUniverse, UniverseManifest, UniverseMembership
from stockbot.research.sector_allocator import (
    apply_point_in_time_sector_cap,
    evaluate_sector_cap_challenger,
)


def _manifest():
    return UniverseManifest(
        source="synthetic",
        survivorship_bias_controlled=True,
        includes_delisted_securities=True,
        point_in_time_membership=True,
    )


def _bars(symbols=("AAA", "AAB", "BBB", "BBC"), periods=80):
    rng = np.random.default_rng(20260909)
    dates = pd.date_range("2025-01-02", periods=periods, freq="B", tz="UTC")
    rows = []
    for j, symbol in enumerate(symbols):
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003 + j * 0.00003, 0.007, periods)))
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "timestamp": dt,
                    "symbol": symbol,
                    "open": float(close[i]),
                    "high": float(close[i] * 1.005),
                    "low": float(close[i] * 0.995),
                    "close": float(close[i]),
                    "volume": float(1_000_000 + 50_000 * j + 1_000 * (i % 13)),
                }
            )
    return pd.DataFrame(rows), dates


def _predictions(dates):
    mapping = {}
    for i, dt in enumerate(dates):
        mapping[(dt, "AAA")] = 4.0 + 0.01 * i
        mapping[(dt, "AAB")] = 3.0 + 0.01 * i
        mapping[(dt, "BBB")] = 0.8 + 0.005 * i
        mapping[(dt, "BBC")] = 0.6 + 0.005 * i
    return pd.Series(
        mapping,
        index=pd.MultiIndex.from_tuples(mapping.keys(), names=["timestamp", "symbol"]),
        dtype=float,
    )


def test_sector_cap_uses_signal_date_classification_not_next_execution_date():
    signal_date = pd.Timestamp("2025-01-03", tz="UTC")
    weights = pd.DataFrame(
        [[0.8, 0.2]],
        index=pd.DatetimeIndex([signal_date], name="timestamp"),
        columns=["AAA", "BBB"],
    )
    universe = PointInTimeUniverse(
        (
            UniverseMembership("AAA", "2020-01-01", "2025-01-06", sector="Technology"),
            UniverseMembership("AAA", "2025-01-06", sector="Financials"),
            UniverseMembership("BBB", "2020-01-01", sector="Technology"),
        ),
        _manifest(),
    )

    constrained, diagnostics = apply_point_in_time_sector_cap(
        weights,
        universe,
        max_sector_weight=0.50,
    )

    # On the signal date both names are Technology, so only 50% can be invested.
    # AAA's next-session Financials classification must not leak backward here.
    assert constrained.loc[signal_date].sum() == pytest.approx(0.50)
    assert diagnostics.loc[signal_date, "max_sector_weight_before"] == pytest.approx(1.0)
    assert diagnostics.loc[signal_date, "max_sector_weight_after"] == pytest.approx(0.50)
    assert diagnostics.loc[signal_date, "cash_weight_after"] == pytest.approx(0.50)


def test_sector_cap_redistributes_excess_without_breaking_cap():
    signal_date = pd.Timestamp("2025-02-03", tz="UTC")
    weights = pd.DataFrame(
        [[0.7, 0.1, 0.2]],
        index=pd.DatetimeIndex([signal_date], name="timestamp"),
        columns=["AAA", "AAB", "BBB"],
    )
    universe = PointInTimeUniverse(
        (
            UniverseMembership("AAA", "2020-01-01", sector="Technology"),
            UniverseMembership("AAB", "2020-01-01", sector="Technology"),
            UniverseMembership("BBB", "2020-01-01", sector="Industrials"),
        ),
        _manifest(),
    )

    constrained, diagnostics = apply_point_in_time_sector_cap(
        weights,
        universe,
        max_sector_weight=0.60,
    )

    tech = constrained.loc[signal_date, ["AAA", "AAB"]].sum()
    industrials = constrained.loc[signal_date, "BBB"]
    assert tech == pytest.approx(0.60)
    assert industrials == pytest.approx(0.40)
    assert constrained.loc[signal_date].sum() == pytest.approx(1.0)
    assert constrained.loc[signal_date, "AAA"] / constrained.loc[signal_date, "AAB"] == pytest.approx(7.0)
    assert diagnostics.loc[signal_date, "max_sector_weight_after"] <= 0.60 + 1e-12
    assert diagnostics.loc[signal_date, "cash_weight_after"] == pytest.approx(0.0)


def test_sector_cap_challenger_reduces_executed_concentration_without_future_sector_leakage():
    bars, dates = _bars()
    predictions = _predictions(dates)
    universe = PointInTimeUniverse(
        (
            UniverseMembership("AAA", "2020-01-01", sector="Technology"),
            UniverseMembership("AAB", "2020-01-01", sector="Technology"),
            UniverseMembership("BBB", "2020-01-01", sector="Industrials"),
            UniverseMembership("BBC", "2020-01-01", sector="Industrials"),
        ),
        _manifest(),
    )

    report = evaluate_sector_cap_challenger(
        bars,
        predictions,
        universe,
        top_fraction=1.0,
        weighting="conviction",
        max_sector_weight=0.60,
        oos_coverage=1.0,
    )

    assert report.max_sector_weight == pytest.approx(0.60)
    assert report.average_signal_max_sector_weight_after <= 0.60 + 1e-12
    assert report.average_signal_max_sector_weight_after < report.average_signal_max_sector_weight_before
    assert report.constrained_sector_exposure.average_max_sector_weight < report.baseline_sector_exposure.average_max_sector_weight
    assert report.constrained_sector_exposure.worst_max_sector_weight <= 0.60 + 1e-12
    assert report.average_cash_weight < 0.05
    assert np.isfinite(report.baseline_score)
    assert np.isfinite(report.constrained_score)
    assert np.isfinite(report.constrained_metrics["sharpe"])
    assert len(report.constrained_net_returns) > 0


def test_sector_cap_fails_closed_when_selected_signal_weight_lacks_sector_history():
    signal_date = pd.Timestamp("2025-02-03", tz="UTC")
    weights = pd.DataFrame(
        [[0.5, 0.5]],
        index=pd.DatetimeIndex([signal_date], name="timestamp"),
        columns=["AAA", "BBB"],
    )
    universe = PointInTimeUniverse(
        (UniverseMembership("AAA", "2020-01-01", sector="Technology"),),
        _manifest(),
    )

    with pytest.raises(ValueError, match="signal-date sector coverage"):
        apply_point_in_time_sector_cap(
            weights,
            universe,
            max_sector_weight=0.60,
            min_sector_coverage=0.90,
        )
