from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from stockbot.data.schemas import DatasetMetadata


class LocalFrameProvider:
    def __init__(self, bars: pd.DataFrame, metadata: DatasetMetadata) -> None:
        self._bars = bars.copy()
        self._metadata = metadata

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    def load_bars(self, symbols: Sequence[str], start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
        result = self._bars.loc[self._bars["symbol"].isin(list(symbols))].copy()
        timestamps = pd.to_datetime(result["timestamp"], utc=True)
        if start is not None:
            result = result.loc[timestamps >= pd.Timestamp(start)]
            timestamps = pd.to_datetime(result["timestamp"], utc=True)
        if end is not None:
            result = result.loc[timestamps <= pd.Timestamp(end)]
        return result.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
