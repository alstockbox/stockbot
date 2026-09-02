from __future__ import annotations

from typing import Protocol, Sequence

import pandas as pd

from stockbot.data.schemas import DatasetMetadata


class MarketDataProvider(Protocol):
    @property
    def metadata(self) -> DatasetMetadata: ...

    def load_bars(self, symbols: Sequence[str], start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame: ...
