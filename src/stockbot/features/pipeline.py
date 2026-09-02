from __future__ import annotations

import pandas as pd

from .technical import technical_features


def build_technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.index.is_monotonic_increasing:
        raise ValueError("market data index must be monotonic increasing")
    if frame.index.has_duplicates:
        raise ValueError("market data index must not contain duplicates")
    return technical_features(frame)
