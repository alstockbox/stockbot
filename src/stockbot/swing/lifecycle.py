"""Explicit, offline cash-only delisting settlement inputs.

Never infer a last-price sale or distribute unverifiable stock consideration.
No broker communication, deposits, positions or order instructions.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from math import isfinite
import re
from typing import Any
from zoneinfo import ZoneInfo

from stockbot.swing.data_quality import _parse_utc


@dataclass(frozen=True)
class DelistingSettlement:
    symbol: str
    last_session: date
    cash_per_share_sek: float
    pay_date: date
    known_at_utc: str
    source_reference: str
    currency: str = "SEK"
    settlement_kind: str = "CASH_ONLY"

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", self.symbol):
            raise ValueError("invalid delisting symbol")
        if not isinstance(self.last_session, date) or isinstance(self.last_session, datetime):
            raise ValueError("last_session must be date")
        if not isinstance(self.pay_date, date) or isinstance(self.pay_date, datetime):
            raise ValueError("pay_date must be date")
        if self.pay_date <= self.last_session:
            raise ValueError("cash settlement must pay after final session")
        if (not isinstance(self.cash_per_share_sek, (int, float))
                or isinstance(self.cash_per_share_sek, bool)
                or not isfinite(self.cash_per_share_sek) or self.cash_per_share_sek < 0):
            raise ValueError("settlement must have nonnegative finite cash per share")
        if self.currency != "SEK" or self.settlement_kind != "CASH_ONLY":
            raise ValueError("only explicit SEK cash delistings are supported")
        if not isinstance(self.source_reference, str) or not self.source_reference.strip():
            raise ValueError("settlement evidence reference required")
        known = _parse_utc(self.known_at_utc)
        last_open = datetime.combine(
            self.last_session, time(9), tzinfo=ZoneInfo("Europe/Stockholm")
        ).astimezone(known.tzinfo)
        # The engine avoids buying on the last tradable day. This guard
        # MUST be known before the opening auction to avoid lookahead.
        if known >= last_open:
            raise ValueError("settlement not known before final session open")


def parse_settlements(payload: Any) -> tuple[DelistingSettlement, ...]:
    if not isinstance(payload, list) or len(payload) > 5000:
        raise ValueError("delisting settlements must be a JSON list (max 5000)")
    expected = set(DelistingSettlement.__dataclass_fields__)
    required = expected - {"currency", "settlement_kind"}
    output: list[DelistingSettlement] = []
    for item in payload:
        if not isinstance(item, dict) or not required.issubset(item) or set(item) - expected:
            raise ValueError("invalid delisting settlement schema")
        data = item.copy()
        try:
            data["last_session"] = date.fromisoformat(data["last_session"])
            data["pay_date"] = date.fromisoformat(data["pay_date"])
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("invalid settlement dates") from exc
        output.append(DelistingSettlement(**data))
    if len({e.symbol for e in output}) != len(output):
        raise ValueError("duplicate delisting settlement symbol")
    return tuple(output)


def match_settlements(
    records: tuple[object, ...],
    settlements: tuple[DelistingSettlement, ...],
    *,
    first_session: date,
    last_session: date,
) -> dict[str, DelistingSettlement]:
    """Every delisting ending inside sample needs documented cash settlement.

    Any noncash or missing payout blocks replay even when backtest did not
    own the affected shares. This avoids silently removing bad performers.
    """
    mapped = {e.symbol: e for e in settlements}
    if len(mapped) != len(settlements):
        raise ValueError("duplicate delisting settlement symbol")
    expected: dict[str, date] = {}
    all_symbols = set()
    for record in records:
        all_symbols.add(record.symbol)
        if (record.status == "DELISTED"
                and first_session <= record.last_session <= last_session):
            expected[record.symbol] = record.last_session
    if set(mapped) != set(expected):
        raise ValueError("missing or unexpected delisting cash settlement")
    if any(e.last_session != expected[e.symbol] for e in settlements):
        raise ValueError("settlement last session mismatches universe")
    if any(e.symbol not in all_symbols for e in settlements):
        raise ValueError("unknown settlement symbol")
    return mapped
