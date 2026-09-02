from __future__ import annotations

import math

import numpy as np
import pandas as pd


def technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"close", "high", "low", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required market columns: {sorted(missing)}")

    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)
    ret1 = close.pct_change()

    out = pd.DataFrame(index=frame.index)
    out["return_1"] = ret1
    out["momentum_5"] = close.pct_change(5)
    out["momentum_20"] = close.pct_change(20)

    sma20 = close.rolling(20, min_periods=20).mean()
    out["sma_20_dist"] = close / sma20 - 1.0

    out["realized_vol_20"] = ret1.rolling(20, min_periods=20).std(ddof=0) * math.sqrt(252)

    vol_mean = volume.rolling(20, min_periods=20).mean()
    vol_std = volume.rolling(20, min_periods=20).std(ddof=0).replace(0, np.nan)
    out["volume_z_20"] = (volume - vol_mean) / vol_std

    prior_high = frame["high"].astype(float).shift(1).rolling(20, min_periods=20).max()
    out["breakout_20"] = close / prior_high - 1.0

    return out.replace([np.inf, -np.inf], np.nan)
