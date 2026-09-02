from __future__ import annotations

from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    name: str

    def generate_signal(self, features: pd.Series) -> float:
        ...
