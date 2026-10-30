"""Offline, synthetic-only XSTO session journal for forward paper *rehearsal*.

This module is a provenance/clock/integrity GATE, not a source of market data
or a trading system. All external input is blocked until independent legal
and economic review; it never calls a broker or starts paper-forward trading.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import fcntl
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4

import exchange_calendars as xcals
import pandas as pd

GENESIS = "0" * 64
VERSION = 1
MAX_SYMBOLS = 5000


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("observed_at_utc and clock_utc must be timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _parse_utc(raw: str) -> datetime:
    if not isinstance(raw, str):
        raise ValueError("timestamp must be a UTC ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid UTC ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("timestamp must explicitly be UTC")
    return parsed


@dataclass(frozen=True)
class PaperBar:
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", self.symbol):
            raise ValueError("invalid paper symbol")
        values = (self.open, self.high, self.low, self.close)
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not isfinite(x) or x <= 0 for x in values):
            raise ValueError("paper prices must be positive finite numbers")
        if self.high < max(self.open, self.low, self.close) or self.low > min(self.open, self.high, self.close):
            raise ValueError("inconsistent OHLC paper bar")
        if (isinstance(self.volume, bool) or not isinstance(self.volume, (int, float))
                or not isfinite(self.volume) or self.volume < 0 or int(self.volume) != self.volume):
            raise ValueError("paper volume must be a nonnegative integer")


@dataclass(frozen=True)
class PaperSession:
    session: date
    observed_at_utc: datetime
    source_kind: str
    source_reference: str
    bars: tuple[PaperBar, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.session, date) or isinstance(self.session, datetime):
            raise ValueError("session must be a date")
        if self.source_kind != "SYNTHETIC_TEST_ONLY":
            raise ValueError("EXTERNAL_MARKET_DATA_UNLICENSED_OR_UNVERIFIED")
        if not isinstance(self.source_reference, str) or not self.source_reference.strip():
            raise ValueError("source reference required")
        if not self.source_reference.startswith("synthetic:"):
            raise ValueError("synthetic source must be identified as synthetic:")
        if not self.bars or len(self.bars) > MAX_SYMBOLS or any(not isinstance(b, PaperBar) for b in self.bars):
            raise ValueError("paper session must contain 1..5000 typed bars")
        symbols = [b.symbol for b in self.bars]
        if len(set(symbols)) != len(symbols):
            raise ValueError("duplicate paper session symbols")
        observed = _utc(self.observed_at_utc)
        calendar = xcals.get_calendar("XSTO")
        trade_day = pd.Timestamp(self.session)
        if not calendar.is_session(trade_day):
            raise ValueError("non-XSTO market session")
        ready_at = calendar.session_close(trade_day).to_pydatetime() + timedelta(minutes=15)
        if observed < ready_at:
            raise ValueError("daily paper bars not available at declared observation time")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PaperSession":
        if not isinstance(data, dict) or set(data) != {
            "session", "observed_at_utc", "source_kind", "source_reference", "bars"
        }:
            raise ValueError("invalid paper session schema")
        try:
            day = date.fromisoformat(data["session"])
            stamp = _parse_utc(data["observed_at_utc"])
            if not isinstance(data["bars"], list):
                raise ValueError("paper bars must be list")
            allowed = {"symbol", "open", "high", "low", "close", "volume"}
            rows = []
            for entry in data["bars"]:
                if not isinstance(entry, dict) or set(entry) != allowed:
                    raise ValueError("invalid paper bar schema")
                rows.append(PaperBar(**entry))
            return cls(day, stamp, data["source_kind"], data["source_reference"], tuple(rows))
        except (TypeError, AttributeError, KeyError) as exc:
            raise ValueError("invalid paper session fields") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session.isoformat(),
            "observed_at_utc": _utc(self.observed_at_utc).isoformat(),
            "source_kind": self.source_kind,
            "source_reference": self.source_reference,
            "bars": [
                {"symbol": b.symbol, "open": b.open, "high": b.high, "low": b.low,
                 "close": b.close, "volume": int(b.volume)}
                for b in sorted(self.bars, key=lambda b: b.symbol)
            ],
        }


def _digest(sequence: int, prior: str, data: dict[str, Any]) -> str:
    content = json.dumps(
        {"schema_version": VERSION, "sequence": sequence, "previous_hash": prior, "session": data},
        sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(content).hexdigest()


@dataclass(frozen=True)
class PaperFeedStatus:
    mode: str
    sessions_staged: int
    last_session: str | None
    chain_head_sha256: str
    symbols_on_last_session: int
    actual_market_data_rights_verified: bool = False
    real_forward_paper_started: bool = False
    broker_orders_sent: int = 0


@dataclass(frozen=True)
class PaperFeedReadiness:
    mode: str
    sessions_staged: int
    complete_xsto_calendar: bool
    constant_symbol_universe: bool
    technical_warmup_complete: bool
    real_market_data_rights_verified: bool = False
    real_forward_paper_allowed: bool = False
    orders_sent: int = 0


class PaperFeedJournal:
    """Single-writer, integrity-checked feed *rehearsal* journal.

    The file is never a source of order instructions. A batch is one daily
    market session. No partial multi-symbol session or same-day rewrite.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock = self.path.with_name(self.path.name + ".lock")

    def _replay(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        prior = GENESIS
        previous_day: date | None = None
        try:
            for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != {
                    "schema_version", "sequence", "previous_hash", "entry_hash", "session"
                }:
                    raise ValueError("invalid paper journal record schema")
                if row["schema_version"] != VERSION or row["sequence"] != number or row["previous_hash"] != prior:
                    raise ValueError("paper journal sequence mismatch")
                data = row["session"]
                session = PaperSession.from_dict(data)
                if session.to_dict() != data:
                    raise ValueError("noncanonical paper journal record")
                if previous_day is not None and session.session <= previous_day:
                    raise ValueError("paper session order/duplicate violated")
                expected = _digest(number, prior, data)
                if row["entry_hash"] != expected:
                    raise ValueError("paper journal tamper/hash mismatch")
                previous_day = session.session
                prior = expected
                records.append(row)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, KeyError) as exc:
            raise ValueError("invalid paper feed journal; fail closed") from exc
        return records

    @staticmethod
    def _status(records: list[dict[str, Any]]) -> PaperFeedStatus:
        return PaperFeedStatus(
            mode="SYNTHETIC_PAPER_REHEARSAL_ONLY",
            sessions_staged=len(records),
            last_session=records[-1]["session"]["session"] if records else None,
            chain_head_sha256=records[-1]["entry_hash"] if records else GENESIS,
            symbols_on_last_session=len(records[-1]["session"]["bars"]) if records else 0,
        )

    def snapshot(self) -> PaperFeedStatus:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            try:
                return self._status(self._replay())
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def readiness(self, *, required_sessions: int = 205) -> PaperFeedReadiness:
        """Structural fixed-universe indicator readiness, NOT trading approval."""
        if not isinstance(required_sessions, int) or isinstance(required_sessions, bool) or required_sessions < 205:
            raise ValueError("indicator warmup requires at least 205 XSTO sessions")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            try:
                entries = self._replay()
                complete = False
                same_universe = False
                if entries:
                    days = pd.DatetimeIndex(
                        [row["session"]["session"] for row in entries]
                    )
                    cal = xcals.get_calendar("XSTO")
                    expected = pd.DatetimeIndex(cal.sessions_in_range(
                        days[0], days[-1]
                    )).tz_localize(None).normalize()
                    complete = days.equals(expected)
                    first_symbols = set(
                        item["symbol"] for item in entries[0]["session"]["bars"]
                    )
                    same_universe = all(
                        {item["symbol"] for item in row["session"]["bars"]} == first_symbols
                        for row in entries
                    )
                return PaperFeedReadiness(
                    mode="SYNTHETIC_TEST_READINESS_ONLY",
                    sessions_staged=len(entries),
                    complete_xsto_calendar=complete,
                    constant_symbol_universe=same_universe,
                    technical_warmup_complete=(
                        complete and same_universe and len(entries) >= required_sessions
                    ),
                )
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def stage(self, event: PaperSession, *, clock_utc: datetime) -> PaperFeedStatus:
        """Persist a synthetic daily session after it is known, without orders."""
        now = _utc(clock_utc)
        if _utc(event.observed_at_utc) > now:
            raise ValueError("paper observation is in the future")
        payload = event.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                entries = self._replay()
                if entries and event.session <= date.fromisoformat(entries[-1]["session"]["session"]):
                    raise ValueError("DUPLICATE_OR_OUT_OF_ORDER_SESSION")
                seq = len(entries) + 1
                prior = entries[-1]["entry_hash"] if entries else GENESIS
                new_row = {
                    "schema_version": VERSION, "sequence": seq, "previous_hash": prior,
                    "entry_hash": _digest(seq, prior, payload), "session": payload
                }
                # Atomic replace prevents a crashed append from creating a
                # silently truncated last line; hash chain still catches edits.
                serialized = "".join(json.dumps(
                    row, sort_keys=True, separators=(",", ":"), allow_nan=False
                ) + "\n" for row in (*entries, new_row))
                tmp = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
                try:
                    with tmp.open("x", encoding="utf-8") as fh:
                        fh.write(serialized)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp, self.path)
                    directory_fd = os.open(self.path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                finally:
                    tmp.unlink(missing_ok=True)
                return self._status([*entries, new_row])
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
