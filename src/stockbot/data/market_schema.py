from __future__ import annotations

import math

import pandas as pd

CANONICAL_BAR_COLUMNS = (
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adj_volume",
    "div_cash",
    "split_factor",
    "provider",
    "retrieved_at",
)


def validate_canonical_bars(frame: pd.DataFrame) -> None:
    missing = [column for column in CANONICAL_BAR_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing canonical bar columns: {missing}")
    if frame.empty:
        raise ValueError("canonical bars cannot be empty")
    if frame[["symbol", "timestamp"]].duplicated().any():
        raise ValueError("duplicate symbol/timestamp observations")

    raw = frame[["open", "high", "low", "close", "volume"]].astype(float)
    if not raw.apply(lambda column: column.map(math.isfinite)).all().all():
        raise ValueError("raw OHLCV must be finite")
    if (raw[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (raw["volume"] < 0).any():
        raise ValueError("volume cannot be negative")
    if (raw["high"] < raw[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("high must be at least open/low/close")
    if (raw["low"] > raw[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("low must be at most open/high/close")

    split = pd.to_numeric(frame["split_factor"], errors="coerce")
    if split.isna().any() or (split <= 0).any():
        raise ValueError("split_factor must be positive")
    div = pd.to_numeric(frame["div_cash"], errors="coerce")
    if div.isna().any():
        raise ValueError("div_cash must be numeric")
    if frame["symbol"].astype(str).str.strip().eq("").any():
        raise ValueError("symbol cannot be empty")
    if frame["provider"].astype(str).str.strip().eq("").any():
        raise ValueError("provider cannot be empty")
    pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    pd.to_datetime(frame["retrieved_at"], utc=True, errors="raise")
