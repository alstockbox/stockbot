from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math

from stockbot.data.live_models import MarketQuote


class ShadowSide(str, Enum):
    LONG = "long"


@dataclass(frozen=True)
class ShadowPosition:
    decision_id: str
    symbol: str
    side: ShadowSide
    notional_usd: float
    estimated_risk_usd: float
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: datetime


@dataclass(frozen=True)
class ShadowExit:
    decision_id: str
    symbol: str
    exit_price: float
    realized_pnl_usd: float
    reason: str
    closed_at: datetime


class ShadowExecutor:
    def __init__(self) -> None:
        self._positions: dict[str, ShadowPosition] = {}
        self._closed: list[ShadowExit] = []

    @property
    def positions(self) -> tuple[ShadowPosition, ...]:
        return tuple(self._positions.values())

    @property
    def closed(self) -> tuple[ShadowExit, ...]:
        return tuple(self._closed)

    @property
    def open_risk_usd(self) -> float:
        return float(sum(p.estimated_risk_usd for p in self._positions.values()))

    @property
    def gross_notional_usd(self) -> float:
        return float(sum(p.notional_usd for p in self._positions.values()))

    def open_long(
        self,
        *,
        decision_id: str,
        quote: MarketQuote,
        notional_usd: float,
        estimated_risk_usd: float,
        reward_risk: float = 2.0,
    ) -> ShadowPosition:
        if decision_id in self._positions:
            raise ValueError("duplicate shadow decision_id")
        if notional_usd <= 0.0 or estimated_risk_usd <= 0.0:
            raise ValueError("notional_usd and estimated_risk_usd must be positive")
        if estimated_risk_usd >= notional_usd:
            raise ValueError("estimated_risk_usd must be smaller than notional_usd")
        if not math.isfinite(reward_risk) or reward_risk <= 0.0:
            raise ValueError("reward_risk must be positive and finite")

        entry = float(quote.ask)
        quantity = float(notional_usd) / entry
        risk_fraction = float(estimated_risk_usd) / float(notional_usd)
        stop_loss = entry * (1.0 - risk_fraction)
        take_profit = entry * (1.0 + reward_risk * risk_fraction)

        position = ShadowPosition(
            decision_id=decision_id,
            symbol=quote.symbol,
            side=ShadowSide.LONG,
            notional_usd=float(notional_usd),
            estimated_risk_usd=float(estimated_risk_usd),
            quantity=quantity,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opened_at=quote.timestamp,
        )
        self._positions[decision_id] = position
        return position

    def unrealized_pnl(self, decision_id: str, quote: MarketQuote) -> float:
        position = self._positions[decision_id]
        if quote.symbol != position.symbol:
            raise ValueError("quote symbol mismatch")
        return float(position.quantity * (float(quote.bid) - position.entry_price))

    def maybe_close(self, decision_id: str, quote: MarketQuote) -> ShadowExit | None:
        position = self._positions[decision_id]
        if quote.symbol != position.symbol:
            raise ValueError("quote symbol mismatch")
        bid = float(quote.bid)
        reason: str | None = None
        if bid <= position.stop_loss:
            reason = "STOP_LOSS"
        elif bid >= position.take_profit:
            reason = "TAKE_PROFIT"
        if reason is None:
            return None
        return self.close(decision_id, quote, reason=reason)

    def close(self, decision_id: str, quote: MarketQuote, *, reason: str) -> ShadowExit:
        position = self._positions.pop(decision_id)
        if quote.symbol != position.symbol:
            self._positions[decision_id] = position
            raise ValueError("quote symbol mismatch")
        realized = float(position.quantity * (float(quote.bid) - position.entry_price))
        result = ShadowExit(
            decision_id=decision_id,
            symbol=position.symbol,
            exit_price=float(quote.bid),
            realized_pnl_usd=realized,
            reason=reason,
            closed_at=quote.timestamp,
        )
        self._closed.append(result)
        return result
