from types import SimpleNamespace

import numpy as np
import pandas as pd

from stockbot.research.stacking import evaluate_oos_stacking


def _bars_and_candidates():
    rng = np.random.default_rng(20260908)
    dates = pd.date_range("2024-01-02", periods=220, freq="B", tz="UTC")
    symbols = ("AAA", "BBB", "CCC", "DDD")
    rows = []
    prediction_maps = [dict(), dict(), dict()]

    market = rng.normal(0.0002, 0.004, len(dates))
    for j, symbol in enumerate(symbols):
        signal = 0.0008 * np.sin(np.arange(len(dates)) / (7.0 + j))
        noise = rng.normal(0.0, 0.003, len(dates))
        returns = market + signal + noise
        close = 100.0 * np.exp(np.cumsum(returns))
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": float(close[i]),
                    "high": float(close[i] * 1.005),
                    "low": float(close[i] * 0.995),
                    "close": float(close[i]),
                    "volume": 1_000_000 + 10_000 * j,
                }
            )
            prediction_maps[0][(symbol, dt)] = float(signal[i] + rng.normal(0.0, 0.0004))
            prediction_maps[1][(symbol, dt)] = float(0.7 * signal[i] + rng.normal(0.0, 0.0007))
            prediction_maps[2][(symbol, dt)] = float(-0.2 * signal[i] + rng.normal(0.0, 0.0010))

    candidates = []
    for index, mapping in enumerate(prediction_maps):
        multi_index = pd.MultiIndex.from_tuples(mapping.keys(), names=["symbol", "timestamp"])
        predictions = pd.Series(list(mapping.values()), index=multi_index, dtype=float)
        candidates.append(
            SimpleNamespace(
                experiment_id=f"candidate-{index}",
                horizon=5,
                result=SimpleNamespace(predictions=predictions),
            )
        )
    return pd.DataFrame(rows), candidates


def test_oos_stacking_produces_purged_meta_predictions_and_metrics():
    bars, candidates = _bars_and_candidates()
    report = evaluate_oos_stacking(bars, candidates, horizon=5)

    assert report.horizon == 5
    assert len(report.member_ids) == 3
    assert report.oos_coverage > 0.0
    assert report.predictions.notna().sum() > 0
    assert np.isfinite(report.score)
    assert np.isfinite(report.metrics["sharpe"])
    assert len(report.net_returns) > 0


def test_oos_stacking_requires_two_same_horizon_members():
    bars, candidates = _bars_and_candidates()
    try:
        evaluate_oos_stacking(bars, candidates[:1], horizon=5)
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("expected ValueError")
