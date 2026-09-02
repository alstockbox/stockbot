from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedWalkForwardSplitter:
    train_periods: int
    test_periods: int
    label_horizon: int
    embargo_periods: int = 0

    def __post_init__(self) -> None:
        for name in ("train_periods", "test_periods", "label_horizon"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.embargo_periods < 0:
            raise ValueError("embargo_periods cannot be negative")

    def split(self, index: pd.Index) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if isinstance(index, pd.MultiIndex):
            if "timestamp" not in index.names:
                raise ValueError("MultiIndex must include timestamp")
            row_dates = pd.DatetimeIndex(index.get_level_values("timestamp"))
        else:
            row_dates = pd.DatetimeIndex(index)
        unique_dates = pd.DatetimeIndex(row_dates.unique()).sort_values()
        test_start = self.train_periods + self.label_horizon
        while test_start + self.test_periods <= len(unique_dates):
            train_end = test_start - self.label_horizon
            train_start = train_end - self.train_periods
            if train_start < 0:
                break
            train_dates = unique_dates[train_start:train_end]
            test_dates = unique_dates[test_start:test_start + self.test_periods]
            yield np.flatnonzero(row_dates.isin(train_dates)), np.flatnonzero(row_dates.isin(test_dates))
            test_start += self.test_periods + self.embargo_periods
