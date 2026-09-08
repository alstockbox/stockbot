from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_by_symbol(
    values: pd.Series,
    symbol: pd.Index,
    window: int,
    operation: str,
) -> pd.Series:
    grouped = values.groupby(symbol)
    if operation == "mean":
        return grouped.transform(lambda s: s.rolling(window, min_periods=window).mean())
    if operation == "std":
        return grouped.transform(lambda s: s.rolling(window, min_periods=window).std(ddof=0))
    if operation == "max":
        return grouped.transform(lambda s: s.rolling(window, min_periods=window).max())
    if operation == "min":
        return grouped.transform(lambda s: s.rolling(window, min_periods=window).min())
    raise ValueError(f"unsupported rolling operation: {operation}")


def _zscore_by_symbol(values: pd.Series, symbol: pd.Index, window: int) -> pd.Series:
    mean = _rolling_by_symbol(values, symbol, window, "mean")
    std = _rolling_by_symbol(values, symbol, window, "std")
    return (values - mean) / std.replace(0.0, np.nan)


def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add causal multi-horizon time-series and same-date cross-sectional features."""

    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names != ["timestamp", "symbol"]:
        raise ValueError("panel must be indexed by timestamp,symbol")

    result = panel.copy().sort_index()
    symbol = result.index.get_level_values("symbol")
    close = result["close"].astype(float)
    open_ = result["open"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    volume = result["volume"].astype(float)

    result["return_1"] = close.groupby(symbol).pct_change()
    result["return_5"] = close.groupby(symbol).pct_change(5)
    result["return_20"] = close.groupby(symbol).pct_change(20)
    result["momentum_5"] = close.groupby(symbol).transform(lambda s: s / s.shift(5) - 1.0)
    result["momentum_20"] = close.groupby(symbol).transform(lambda s: s / s.shift(20) - 1.0)
    result["momentum_60"] = close.groupby(symbol).transform(lambda s: s / s.shift(60) - 1.0)

    for window in (5, 20, 60):
        result[f"realized_vol_{window}"] = (
            _rolling_by_symbol(result["return_1"], symbol, window, "std") * np.sqrt(252.0)
        )

    for window in (5, 20, 60):
        result[f"volume_z_{window}"] = _zscore_by_symbol(volume, symbol, window)
    dollar_volume = close * volume
    result["log_dollar_volume"] = np.log1p(dollar_volume.clip(lower=0.0))
    result["dollar_volume_z_20"] = _zscore_by_symbol(result["log_dollar_volume"], symbol, 20)

    for window in (5, 20, 60):
        sma = _rolling_by_symbol(close, symbol, window, "mean")
        result[f"sma_{window}_dist"] = close / sma.replace(0.0, np.nan) - 1.0
    rolling_high_20 = _rolling_by_symbol(high, symbol, 20, "max")
    rolling_low_20 = _rolling_by_symbol(low, symbol, 20, "min")
    result["distance_high_20"] = close / rolling_high_20.replace(0.0, np.nan) - 1.0
    result["distance_low_20"] = close / rolling_low_20.replace(0.0, np.nan) - 1.0

    result["range_1"] = (high - low) / close.replace(0.0, np.nan)
    result["range_20"] = _rolling_by_symbol(result["range_1"], symbol, 20, "mean")
    previous_close = close.groupby(symbol).shift(1)
    result["gap_1"] = open_ / previous_close.replace(0.0, np.nan) - 1.0

    gains = result["return_1"].clip(lower=0.0)
    losses = -result["return_1"].clip(upper=0.0)
    avg_gain = _rolling_by_symbol(gains, symbol, 14, "mean")
    avg_loss = _rolling_by_symbol(losses, symbol, 14, "mean")
    result["rsi_14"] = avg_gain / (avg_gain + avg_loss).replace(0.0, np.nan)

    result["momentum_vol_ratio_20"] = (
        result["momentum_20"] / result["realized_vol_20"].replace(0.0, np.nan)
    )
    result["momentum_volume_interaction"] = result["momentum_20"] * result["volume_z_20"]

    rank_sources = {
        "return_1_rank": "return_1",
        "momentum_5_rank": "momentum_5",
        "momentum_rank": "momentum_20",
        "momentum_60_rank": "momentum_60",
        "volatility_5_rank": "realized_vol_5",
        "volatility_rank": "realized_vol_20",
        "volatility_60_rank": "realized_vol_60",
        "volume_rank": "volume_z_20",
        "liquidity_rank": "log_dollar_volume",
        "range_rank": "range_20",
        "rsi_rank": "rsi_14",
    }
    timestamp_group = result.groupby(level="timestamp")
    for target, source in rank_sources.items():
        result[target] = timestamp_group[source].rank(pct=True, method="average")

    return result.replace([np.inf, -np.inf], np.nan)
