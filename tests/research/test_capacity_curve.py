import numpy as np
import pandas as pd

from stockbot.research.capacity_curve import evaluate_capacity_curve
from stockbot.research.liquidity_execution import LiquidityExecutionConfig


def _bars_and_predictions():
    dates = pd.date_range("2025-01-02", periods=90, freq="B", tz="UTC")
    rows = []
    predictions = {}
    for symbol, drift, signal in (("AAA", 0.0015, 0.02), ("BBB", 0.0001, 0.005)):
        price = 100.0
        for dt in dates:
            price *= 1.0 + drift
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": price,
                    "high": price * 1.005,
                    "low": price * 0.995,
                    "close": price,
                    "volume": 50_000,
                }
            )
            predictions[(symbol, dt)] = signal
    index = pd.MultiIndex.from_tuples(predictions, names=["symbol", "timestamp"])
    prediction_series = pd.Series(list(predictions.values()), index=index, dtype=float)
    return pd.DataFrame(rows), prediction_series


def test_capacity_curve_detects_liquidity_degradation_as_capital_grows():
    bars, predictions = _bars_and_predictions()
    base = LiquidityExecutionConfig(
        capital=10_000.0,
        commission_bps=1.0,
        spread_bps=4.0,
        impact_bps_at_one_pct_adv=10.0,
        max_participation=0.01,
        adv_window=5,
    )
    report = evaluate_capacity_curve(
        bars,
        predictions,
        capital_levels=(10_000.0, 100_000.0, 1_000_000.0, 5_000_000.0),
        base_config=base,
        min_score_retention=0.0 + 1e-9,
        max_partial_fill_fraction=0.50,
        max_tracking_error=1.0,
        min_sharpe=-100.0,
    )

    assert [point.capital for point in report.points] == [10_000.0, 100_000.0, 1_000_000.0, 5_000_000.0]
    assert report.points[-1].partial_fill_fraction >= report.points[0].partial_fill_fraction
    assert report.points[-1].average_tracking_error >= report.points[0].average_tracking_error
    assert report.points[0].score_retention > 0.0
    assert 0.0 <= report.capacity_score <= 1.0


def test_capacity_curve_rejects_invalid_capital_levels():
    bars, predictions = _bars_and_predictions()
    try:
        evaluate_capacity_curve(bars, predictions, capital_levels=(10_000.0, 0.0))
    except ValueError as exc:
        assert "capital_levels" in str(exc)
    else:
        raise AssertionError("expected ValueError")
