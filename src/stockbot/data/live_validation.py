from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from stockbot.data.live_models import AccountSnapshot, MarketQuote, SymbolSpec


@dataclass(frozen=True)
class LiveValidationConfig:
    max_quote_age_seconds: float = 30.0
    max_account_age_seconds: float = 120.0
    max_spread_fraction: float = 0.02

    def __post_init__(self) -> None:
        if self.max_quote_age_seconds <= 0.0:
            raise ValueError("max_quote_age_seconds must be positive")
        if self.max_account_age_seconds <= 0.0:
            raise ValueError("max_account_age_seconds must be positive")
        if not 0.0 < self.max_spread_fraction < 1.0:
            raise ValueError("max_spread_fraction must be in (0,1)")


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reasons: tuple[str, ...] = ()


def validate_quote(
    quote: MarketQuote,
    *,
    now: datetime,
    config: LiveValidationConfig | None = None,
) -> ValidationResult:
    config = config or LiveValidationConfig()
    reasons: list[str] = []
    if quote.age_seconds(now) > config.max_quote_age_seconds:
        reasons.append("STALE_QUOTE")
    if quote.spread_fraction > config.max_spread_fraction:
        reasons.append("EXCESSIVE_SPREAD")
    return ValidationResult(not reasons, tuple(reasons))


def validate_account(
    account: AccountSnapshot,
    *,
    now: datetime,
    config: LiveValidationConfig | None = None,
) -> ValidationResult:
    config = config or LiveValidationConfig()
    age = max(0.0, (now - account.timestamp).total_seconds())
    reasons: list[str] = []
    if age > config.max_account_age_seconds:
        reasons.append("STALE_ACCOUNT")
    if account.equity <= 0.0:
        reasons.append("INVALID_EQUITY")
    if account.free_margin < 0.0:
        reasons.append("INVALID_FREE_MARGIN")
    return ValidationResult(not reasons, tuple(reasons))


def validate_symbol(spec: SymbolSpec) -> ValidationResult:
    reasons: list[str] = []
    if not spec.trade_enabled:
        reasons.append("SYMBOL_DISABLED")
    if spec.tick_size <= 0.0:
        reasons.append("INVALID_TICK_SIZE")
    if spec.volume_min <= 0.0 or spec.volume_step <= 0.0:
        reasons.append("INVALID_VOLUME_RULES")
    if spec.volume_max < spec.volume_min:
        reasons.append("INVALID_VOLUME_RANGE")
    return ValidationResult(not reasons, tuple(reasons))
