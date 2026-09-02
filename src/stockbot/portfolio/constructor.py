from __future__ import annotations

from dataclasses import dataclass
import math

from stockbot.domain.models import TargetPosition


@dataclass(frozen=True)
class PortfolioConfig:
    target_volatility: float = 0.15
    max_position_weight: float = 1.0
    min_signal: float = 0.0

    def __post_init__(self) -> None:
        if self.target_volatility <= 0:
            raise ValueError("target_volatility must be positive")
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("max_position_weight must be in (0,1]")
        if not 0 <= self.min_signal <= 1:
            raise ValueError("min_signal must be in [0,1]")


def target_from_signal(
    symbol: str,
    signal: float,
    volatility: float,
    config: PortfolioConfig | None = None,
) -> TargetPosition:
    config = config or PortfolioConfig()
    signal = max(0.0, min(1.0, float(signal)))
    if signal < config.min_signal:
        return TargetPosition(symbol=symbol, weight=0.0)
    if not math.isfinite(volatility) or volatility <= 0:
        vol_scalar = 0.0
    else:
        vol_scalar = min(1.5, config.target_volatility / float(volatility))
    weight = min(config.max_position_weight, signal * vol_scalar)
    return TargetPosition(symbol=symbol, weight=max(0.0, weight))
