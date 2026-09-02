import numpy as np
import pandas as pd

from stockbot.features.pipeline import build_technical_features


def _frame(n=80):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(100, 140, n), index=idx)
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.linspace(1_000_000, 2_000_000, n),
    }, index=idx)


def test_feature_pipeline_is_causal_when_future_prices_change():
    original = _frame()
    mutated = original.copy()
    cutoff = original.index[50]
    mutated.loc[mutated.index > cutoff, "close"] *= 4
    a = build_technical_features(original)
    b = build_technical_features(mutated)
    pd.testing.assert_frame_equal(a.loc[:cutoff], b.loc[:cutoff])


def test_feature_pipeline_exposes_core_signal_inputs_without_backfill():
    features = build_technical_features(_frame())
    expected = {"return_1", "momentum_5", "momentum_20", "sma_20_dist", "realized_vol_20", "volume_z_20", "breakout_20"}
    assert expected.issubset(features.columns)
    assert pd.isna(features.iloc[0]["momentum_20"])
