from __future__ import annotations

import math

import pandas as pd


def _finite(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


class TrendStrategy:
    name = "trend"

    def generate_signal(self, features: pd.Series) -> float:
        momentum = _finite(features, "momentum_20")
        distance = _finite(features, "sma_20_dist")
        vol = max(_finite(features, "realized_vol_20", 0.20), 0.05)
        strength = (0.65 * momentum + 0.35 * distance) / vol
        return _sigmoid(8.0 * strength)


class MomentumStrategy:
    name = "momentum"

    def generate_signal(self, features: pd.Series) -> float:
        momentum = _finite(features, "momentum_20")
        vol = max(_finite(features, "realized_vol_20", 0.20), 0.05)
        return _sigmoid(7.0 * momentum / vol)


class MeanReversionStrategy:
    name = "mean_reversion"

    def generate_signal(self, features: pd.Series) -> float:
        momentum = _finite(features, "momentum_5")
        vol = max(_finite(features, "realized_vol_20", 0.20), 0.05)
        return _sigmoid(-5.0 * momentum / vol)


class BreakoutStrategy:
    name = "breakout"

    def generate_signal(self, features: pd.Series) -> float:
        breakout = _finite(features, "breakout_20")
        vol = max(_finite(features, "realized_vol_20", 0.20), 0.05)
        return _sigmoid(8.0 * breakout / vol)


BASELINE_STRATEGIES = (
    TrendStrategy(),
    MomentumStrategy(),
    MeanReversionStrategy(),
    BreakoutStrategy(),
)
