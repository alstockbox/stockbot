from __future__ import annotations

import math

import pandas as pd

from stockbot.domain.models import MarketRegime


def _value(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def detect_regime(features: pd.Series) -> MarketRegime:
    momentum = _value(features, "momentum_20")
    distance = _value(features, "sma_20_dist")
    vol = _value(features, "realized_vol_20", 0.20)

    if vol >= 0.55 or (momentum <= -0.04 and distance <= -0.02):
        return MarketRegime.BEAR_STRESS
    if momentum >= 0.03 and distance >= 0.015 and vol < 0.45:
        return MarketRegime.BULL_TREND
    return MarketRegime.NEUTRAL_CHOP
