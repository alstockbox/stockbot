from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class MarketQuote:
    symbol: str
    bid: float
    ask: float
    timestamp: datetime
    source: str = "unknown"

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol is required")
        if not math.isfinite(self.bid) or not math.isfinite(self.ask):
            raise ValueError("bid/ask must be finite")
        if self.bid <= 0.0 or self.ask <= 0.0:
            raise ValueError("bid/ask must be positive")
        if self.ask < self.bid:
            raise ValueError("ask must be >= bid")
        object.__setattr__(self, "timestamp", _aware(self.timestamp))

    @property
    def spread(self) -> float:
        return float(self.ask - self.bid)

    @property
    def midpoint(self) -> float:
        return float((self.ask + self.bid) / 2.0)

    @property
    def spread_fraction(self) -> float:
        midpoint = self.midpoint
        return 0.0 if midpoint <= 0.0 else self.spread / midpoint

    def age_seconds(self, now: datetime) -> float:
        now_utc = _aware(now)
        return max(0.0, (now_utc - self.timestamp).total_seconds())


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    tick_size: float
    tick_value: float
    volume_min: float
    volume_max: float
    volume_step: float
    trade_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol is required")
        for name in ("tick_size", "volume_min", "volume_max", "volume_step"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.tick_value)) or self.tick_value < 0.0:
            raise ValueError("tick_value must be finite and non-negative")
        if self.volume_max < self.volume_min:
            raise ValueError("volume_max must be >= volume_min")


@dataclass(frozen=True)
class AccountSnapshot:
    balance: float
    equity: float
    free_margin: float
    timestamp: datetime
    currency: str = "USD"

    def __post_init__(self) -> None:
        for name in ("balance", "equity", "free_margin"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.balance < 0.0 or self.equity <= 0.0 or self.free_margin < 0.0:
            raise ValueError("account values must be non-negative and equity positive")
        if not self.currency.strip():
            raise ValueError("currency is required")
        object.__setattr__(self, "timestamp", _aware(self.timestamp))


@dataclass(frozen=True)
class MarketBar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol is required")
        prices = (float(self.open), float(self.high), float(self.low), float(self.close))
        if any(not math.isfinite(value) or value <= 0.0 for value in prices):
            raise ValueError("bar prices must be finite and positive")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high is inconsistent")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low is inconsistent")
        if not math.isfinite(float(self.volume)) or self.volume < 0.0:
            raise ValueError("volume must be finite and non-negative")
        object.__setattr__(self, "timestamp", _aware(self.timestamp))
