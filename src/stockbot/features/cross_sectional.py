from __future__ import annotations

import numpy as np
import pandas as pd


def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names != ["timestamp", "symbol"]:
        raise ValueError("panel must be indexed by timestamp,symbol")
    result = panel.copy().sort_index()
    symbol = result.index.get_level_values("symbol")
    close = result["close"].astype(float)
    volume = result["volume"].astype(float)
    result["return_1"] = close.groupby(symbol).pct_change()
    result["momentum_20"] = close.groupby(symbol).transform(lambda s: s / s.shift(20) - 1.0)
    result["realized_vol_20"] = result["return_1"].groupby(symbol).transform(lambda s: s.rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0))
    vol_mean = volume.groupby(symbol).transform(lambda s: s.rolling(20, min_periods=20).mean())
    vol_std = volume.groupby(symbol).transform(lambda s: s.rolling(20, min_periods=20).std(ddof=0))
    result["volume_z_20"] = (volume - vol_mean) / vol_std.replace(0.0, np.nan)
    result["momentum_rank"] = result.groupby(level="timestamp")["momentum_20"].rank(pct=True, method="average")
    result["volatility_rank"] = result.groupby(level="timestamp")["realized_vol_20"].rank(pct=True, method="average")
    result["volume_rank"] = result.groupby(level="timestamp")["volume_z_20"].rank(pct=True, method="average")
    return result
