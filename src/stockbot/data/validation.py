from __future__ import annotations

import pandas as pd

REQUIRED_BAR_COLUMNS = ("symbol", "timestamp", "open", "high", "low", "close", "volume")


def validate_bars(frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_BAR_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing bar columns: {missing}")
    if frame.empty:
        raise ValueError("bars cannot be empty")
    if frame[["symbol", "timestamp"]].duplicated().any():
        raise ValueError("duplicate symbol/timestamp observations")
    if (frame["close"].astype(float) <= 0).any():
        raise ValueError("close must be positive")
    if (frame["volume"].astype(float) < 0).any():
        raise ValueError("volume cannot be negative")
    expected = frame.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
    current = frame.reset_index(drop=True)
    if not current[["symbol", "timestamp"]].equals(expected[["symbol", "timestamp"]]):
        raise ValueError("bars must be sorted by symbol,timestamp")


def as_of_filter(frame: pd.DataFrame, decision_time: pd.Timestamp) -> pd.DataFrame:
    if "available_time" not in frame.columns:
        return frame.copy()
    decision_time = pd.Timestamp(decision_time)
    return frame.loc[pd.to_datetime(frame["available_time"], utc=True) <= decision_time].copy()
