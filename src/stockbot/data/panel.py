from __future__ import annotations

import pandas as pd

from stockbot.data.validation import validate_bars


def build_panel(frame: pd.DataFrame) -> pd.DataFrame:
    if frame[["symbol", "timestamp"]].duplicated().any():
        raise ValueError("duplicate symbol/timestamp observations")
    canonical = frame.copy()
    canonical["timestamp"] = pd.to_datetime(canonical["timestamp"], utc=True)
    canonical = canonical.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
    validate_bars(canonical)
    return canonical.set_index(["timestamp", "symbol"]).sort_index()
