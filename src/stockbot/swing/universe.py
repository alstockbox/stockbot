"""Point-in-time instrument-lifecycle STRUCTURAL audit, without broker access.

Designed to reject survivorship-friendly assumptions and identify where
the existing fully synchronized backtest cannot safely replay IPOs/delistings.
Does NOT establish that the supplied universe includes all vanished names.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from hashlib import sha256
import json
from math import isfinite
import re
from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from stockbot.swing.data_quality import _parse_utc


@dataclass(frozen=True)
class InstrumentHistory:
    symbol: str
    isin: str
    first_session: date
    last_session: date | None
    first_known_at_utc: str
    last_known_at_utc: str | None
    status: str

    def __post_init__(self):
        if not isinstance(self.symbol, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", self.symbol):
            raise ValueError("invalid universe symbol")
        if not isinstance(self.isin, str) or not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{10}", self.isin):
            raise ValueError("invalid ISIN format")
        if not isinstance(self.first_session, date) or isinstance(self.first_session, datetime):
            raise ValueError("first_session must be date")
        if self.status not in ("ACTIVE", "DELISTED"):
            raise ValueError("status must be ACTIVE or DELISTED")
        at = _parse_utc(self.first_known_at_utc)
        listing_open = datetime.combine(self.first_session, time(9), tzinfo=ZoneInfo("Europe/Stockholm"))
        if at > listing_open.astimezone(at.tzinfo):
            raise ValueError("listing was not known before first open")
        if self.last_session is not None:
            if not isinstance(self.last_session, date) or isinstance(self.last_session, datetime):
                raise ValueError("invalid last_session")
            if self.last_session < self.first_session:
                raise ValueError("last_session precedes first_session")
        if self.status == "ACTIVE" and (self.last_session is not None or self.last_known_at_utc is not None):
            raise ValueError("active instrument cannot have delisting metadata")
        if self.status == "DELISTED":
            if self.last_session is None or not self.last_known_at_utc:
                raise ValueError("delisted instrument requires last session and disclosure timestamp")
            _parse_utc(self.last_known_at_utc)


def parse_universe_records(raw: Any) -> tuple[InstrumentHistory, ...]:
    if not isinstance(raw, list) or not (1 <= len(raw) <= 5000):
        raise ValueError("universe JSON must be nonempty list of at most 5000 records")
    expected = set(InstrumentHistory.__dataclass_fields__)
    records = []
    for data in raw:
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("universe record schema is incomplete or unknown")
        entry = data.copy()
        try:
            entry["first_session"] = date.fromisoformat(entry["first_session"])
            if entry["last_session"] is not None:
                entry["last_session"] = date.fromisoformat(entry["last_session"])
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("invalid universe lifecycle date") from exc
        records.append(InstrumentHistory(**entry))
    if len({r.symbol for r in records}) != len(records):
        raise ValueError("duplicate symbol across universe records; versioned symbol mapping required")
    if len({r.isin for r in records}) != len(records):
        raise ValueError("ambiguous ISIN mapping; versioned symbol mapping required")
    return tuple(records)


@dataclass(frozen=True)
class UniverseReceipt:
    mode: str
    data_sha256: str
    universe_sha256: str
    first_session: str
    last_session: str
    symbols_in_bars: int
    active_at_last_session: int
    delisted_symbols_in_bars: int
    listing_transitions_inside_window: int
    delisting_transitions_inside_window: int
    historical_universe_independently_verified: bool
    corporate_action_outcomes_verified: bool
    compatible_with_current_fixed_panel_engine: bool
    authorized_for_paper_or_live: bool

    def to_dict(self):
        return asdict(self)


def audit_instrument_history(
    frame: pd.DataFrame,
    records: tuple[InstrumentHistory, ...],
    *,
    collected_at_utc: str,
) -> UniverseReceipt:
    """Detect missing bars within each listing lifespan and phantom bars outside it.

    Unlike validate_ohlcv, accepts uneven lifecycles for OFFLINE AUDIT ONLY.
    Requires contiguous XSTO sessions where an instrument is tradable; a real
    trading halt should have a zero-volume bar or a documented halt event.
    """
    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError("missing OHLCV universe audit fields")
    if not records:
        raise ValueError("universe record required")
    df = frame[sorted(required)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df.empty or df["date"].isna().any() or (df["date"] != df["date"].dt.normalize()).any():
        raise ValueError("invalid daily XSTO universe dates")
    df["symbol"] = df["symbol"].astype(str)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not df[col].map(isfinite).all():
            raise ValueError(f"nonfinite {col}")
    if (df[["open", "high", "low", "close"]] <= 0).any().any() or (df["volume"] < 0).any():
        raise ValueError("bad OHLCV prices or volume")
    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any() or (
        df["low"] > df[["open", "close"]].min(axis=1)
    ).any():
        raise ValueError("bad OHLC range")
    if df.duplicated(["symbol", "date"]).any():
        raise ValueError("duplicate OHLCV symbol/date")
    first = df["date"].min()
    last = df["date"].max()
    now = _parse_utc(collected_at_utc)
    final_market_close = last.replace(hour=17, minute=30).tz_localize(
        "Europe/Stockholm"
    ).tz_convert("UTC").to_pydatetime()
    if now < final_market_close:
        raise ValueError("universe price data collected before close")
    cal = xcals.get_calendar("XSTO")
    sessions = pd.DatetimeIndex(cal.sessions_in_range(
        first, last
    )).tz_localize(None).normalize()
    all_dates = set(sessions)
    present_dates = pd.DatetimeIndex(df["date"].unique())
    if len(present_dates.difference(sessions)):
        raise ValueError("bars contain non-XSTO date")
    refs = {r.symbol: r for r in records}
    if len(refs) != len(records) or len({r.isin for r in records}) != len(records):
        raise ValueError("duplicate universe symbol/ISIN")
    symbols = set(df["symbol"])
    if symbols - set(refs):
        raise ValueError("OHLCV contains instrument absent from historical universe")
    overlaps = 0
    ended = 0
    included_delisted = 0
    active_last = 0
    fixed_panel_compatible = True
    for record in records:
        if _parse_utc(record.first_known_at_utc) > now:
            raise ValueError("listing disclosure is after collection as-of time")
        if record.last_known_at_utc and _parse_utc(record.last_known_at_utc) > now:
            raise ValueError("delisting disclosure is after collection as-of time")
        start = pd.Timestamp(record.first_session)
        end = pd.Timestamp(record.last_session) if record.last_session else last
        bounded_begin = max(start, first)
        bounded_end = min(end, last)
        if bounded_begin > bounded_end:
            if record.symbol in symbols:
                raise ValueError("phantom bars for out-of-window instrument")
            continue
        if record.symbol not in symbols:
            raise ValueError(f"missing eligible instrument: {record.symbol}")
        if start > first and start <= last:
            overlaps += 1
            fixed_panel_compatible = False
        if record.last_session is not None and first <= end < last:
            ended += 1
            fixed_panel_compatible = False
        if record.status == "DELISTED":
            included_delisted += 1
        if record.status == "ACTIVE" and first <= last:
            active_last += 1
        data = pd.DatetimeIndex(df.loc[df["symbol"] == record.symbol, "date"]).sort_values()
        required_sessions = sessions[(sessions >= bounded_begin) & (sessions <= bounded_end)]
        if len(data.difference(required_sessions)):
            raise ValueError(f"phantom bar before listing or after delisting: {record.symbol}")
        if not data.equals(required_sessions):
            raise ValueError(f"missing tradable-session bars: {record.symbol}")
    if symbols != set(r.symbol for r in records if
                      pd.Timestamp(r.first_session) <= last and
                      (r.last_session is None or pd.Timestamp(r.last_session) >= first)):
        raise ValueError("instrument coverage incomplete")
    canonical = df.sort_values(["date", "symbol"]).to_csv(
        index=False, date_format="%Y-%m-%d", float_format="%.12g"
    ).encode()
    records_json = json.dumps([
        {**asdict(r), "first_session": r.first_session.isoformat(),
         "last_session": r.last_session.isoformat() if r.last_session else None}
        for r in sorted(records, key=lambda r: r.symbol)
    ], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return UniverseReceipt(
        mode="STRUCTURAL_UNIVERSE_AUDIT_ONLY",
        data_sha256=sha256(canonical).hexdigest(),
        universe_sha256=sha256(records_json.encode()).hexdigest(),
        first_session=str(first.date()), last_session=str(last.date()),
        symbols_in_bars=len(symbols),
        active_at_last_session=active_last,
        delisted_symbols_in_bars=included_delisted,
        listing_transitions_inside_window=overlaps,
        delisting_transitions_inside_window=ended,
        historical_universe_independently_verified=False,
        corporate_action_outcomes_verified=False,
        compatible_with_current_fixed_panel_engine=fixed_panel_compatible,
        authorized_for_paper_or_live=False,
    )
