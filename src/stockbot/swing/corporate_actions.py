"""Corporate actions for RAW-price Swedish equity swing research only.

All action records require separately verified market-data provenance.
No broker, network or live trading integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from math import isfinite
import re
from typing import Literal
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class CorporateAction:
    symbol: str
    ex_date: date
    kind: Literal["SPLIT", "DIVIDEND"]
    announced_at: datetime
    numerator: int = 1
    denominator: int = 1
    dividend_per_share_sek: float = 0.0
    pay_date: date | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", self.symbol):
            raise ValueError("invalid action symbol")
        if not isinstance(self.ex_date, date) or isinstance(self.ex_date, datetime):
            raise ValueError("ex_date must be a date")
        if self.kind not in ("SPLIT", "DIVIDEND"):
            raise ValueError("unknown action")
        if not isinstance(self.announced_at, datetime) or self.announced_at.utcoffset() is None:
            raise ValueError("announced_at must have a timezone")
        opening = datetime.combine(self.ex_date, time(hour=9), tzinfo=ZoneInfo("Europe/Stockholm"))
        if self.announced_at >= opening:
            raise ValueError("action must have been announced before ex-date open")
        if self.kind == "SPLIT":
            if (isinstance(self.numerator, bool) or isinstance(self.denominator, bool)
                    or not isinstance(self.numerator, int) or not isinstance(self.denominator, int)
                    or self.numerator <= 0 or self.denominator <= 0
                    or self.numerator == self.denominator):
                raise ValueError("invalid split ratio")
            if self.pay_date is not None or self.dividend_per_share_sek != 0:
                raise ValueError("split must not contain dividend data")
        else:
            if self.numerator != 1 or self.denominator != 1:
                raise ValueError("dividend may not contain split data")
            if (not isinstance(self.dividend_per_share_sek, (int, float))
                    or isinstance(self.dividend_per_share_sek, bool)
                    or not isfinite(self.dividend_per_share_sek)
                    or self.dividend_per_share_sek <= 0):
                raise ValueError("invalid dividend")
            if not isinstance(self.pay_date, date) or isinstance(self.pay_date, datetime):
                raise ValueError("dividend requires pay_date")
            if self.pay_date < self.ex_date:
                raise ValueError("pay_date precedes ex_date")


def validate_actions(
    actions: tuple[CorporateAction, ...],
    *,
    symbols: set[str],
    sessions: set[date],
) -> dict[date, tuple[CorporateAction, ...]]:
    calendar: dict[date, list[CorporateAction]] = {}
    seen: set[tuple[str, date, str]] = set()
    for event in actions:
        if not isinstance(event, CorporateAction):
            raise ValueError("invalid corporate action object")
        if event.symbol not in symbols:
            raise ValueError("action for unknown symbol")
        if event.ex_date not in sessions:
            raise ValueError("action date is not in OHLCV market sessions")
        key = (event.symbol, event.ex_date, event.kind)
        if key in seen:
            raise ValueError("duplicate action for symbol/date")
        seen.add(key)
        calendar.setdefault(event.ex_date, []).append(event)
    return {
        d: tuple(sorted(events, key=lambda e: (e.symbol, e.kind != "SPLIT")))
        for d, events in calendar.items()
    }
