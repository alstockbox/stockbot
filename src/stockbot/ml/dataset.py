from __future__ import annotations

import pandas as pd


def make_forward_return_dataset(
    features: pd.DataFrame,
    close: pd.Series,
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    close = close.astype(float).reindex(features.index)
    target = close.shift(-horizon) / close - 1.0
    joined = features.copy()
    joined["__target__"] = target
    joined = joined.dropna(axis=0, how="any")
    y = joined.pop("__target__")
    return joined, y
