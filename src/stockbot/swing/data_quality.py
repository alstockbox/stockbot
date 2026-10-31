"""Strict Nasdaq Stockholm session and provenance checks for offline OHLCV research.

Never downloads data, grants a license, or makes broker orders. A passing
receipt means structural validation ONLY, not historical alpha or legal rights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Mapping

import exchange_calendars as xcals
import pandas as pd

from stockbot.swing.engine import validate_ohlcv


@dataclass(frozen=True)
class XSTOProvenance:
    provider: str
    terms_reference: str
    collected_at_utc: str
    currency: str = "SEK"
    price_basis: str = "RAW"
    corporate_actions_checked: bool = False
    historical_universe_checked: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "XSTOProvenance":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data).difference(allowed)
        if unknown:
            raise ValueError(f"unknown provenance fields: {sorted(unknown)}")
        try:
            return cls(**data)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("incomplete provenance record") from exc

    def __post_init__(self) -> None:
        if (not isinstance(self.provider, str) or not isinstance(self.terms_reference, str)
                or not self.provider.strip() or not self.terms_reference.strip()):
            raise ValueError("data provider and terms reference are required")
        if self.currency != "SEK":
            raise ValueError("XSTO research requires explicit SEK data")
        if self.price_basis not in ("RAW", "SPLIT_ADJUSTED"):
            raise ValueError("price_basis must be RAW or SPLIT_ADJUSTED")
        if not isinstance(self.corporate_actions_checked, bool) or not isinstance(
            self.historical_universe_checked, bool
        ):
            raise ValueError("verification fields must be boolean")
        _parse_utc(self.collected_at_utc)


def _parse_utc(value: str) -> datetime:
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise ValueError("invalid collected_at_utc") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("collected_at_utc must be timezone-aware")
    if stamp.utcoffset().total_seconds() != 0:
        raise ValueError("collected_at_utc must explicitly use UTC")
    return stamp.astimezone(timezone.utc)


@dataclass(frozen=True)
class XSTOQualityReceipt:
    mode: str
    exchange: str
    data_sha256: str
    manifest_sha256: str
    symbols: int
    sessions: int
    bars: int
    first_session: str
    last_session: str
    license_independently_verified: bool
    corporate_actions_independently_verified: bool
    historical_universe_independently_verified: bool
    research_grade: str
    order_execution_permitted: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_xsto_data(
    frame: pd.DataFrame,
    manifest: XSTOProvenance,
) -> XSTOQualityReceipt:
    """Fail closed for holidays, missing sessions, data timestamps and bad bars.

    The identical-session policy is appropriate for a fixed historical universe
    only; genuine suspensions, listings and delistings must be modeled as events
    before those datasets can be admitted. This does not validate survivorship.
    """
    clean = validate_ohlcv(frame)
    actual = pd.DatetimeIndex(clean["date"].drop_duplicates().sort_values())
    calendar = xcals.get_calendar("XSTO")
    first, last = actual[0], actual[-1]
    try:
        expected = calendar.sessions_in_range(first, last)
    except (ValueError, TypeError) as exc:
        raise ValueError("XSTO calendar cannot cover dataset date range") from exc
    expected = pd.DatetimeIndex(expected).tz_localize(None).normalize()

    invalid = actual.difference(expected)
    missing = expected.difference(actual)
    if len(invalid):
        raise ValueError(f"OHLCV includes non-XSTO session: {invalid[0].date()}")
    if len(missing):
        raise ValueError(f"OHLCV omits XSTO session: {missing[0].date()}")

    collected = _parse_utc(manifest.collected_at_utc)
    # Daily candles are not considered final before the scheduled close; 17:30
    # is conservative for early-close sessions. Never claim tick-level timing.
    last_close = (last + pd.Timedelta(hours=17, minutes=30)).tz_localize(
        "Europe/Stockholm"
    ).tz_convert("UTC")
    if collected < last_close.to_pydatetime():
        raise ValueError("collected_at_utc precedes final daily candle availability")

    # Canonical digest independent of CSV line endings or input row ordering.
    canonical = clean.to_csv(index=False, date_format="%Y-%m-%d",
                             float_format="%.12g").encode("utf-8")
    manifest_json = json.dumps(asdict(manifest), sort_keys=True,
                               separators=(",", ":"), ensure_ascii=False)
    return XSTOQualityReceipt(
        mode="BACKTEST_ONLY",
        exchange="XSTO",
        data_sha256=sha256(canonical).hexdigest(),
        manifest_sha256=sha256(manifest_json.encode("utf-8")).hexdigest(),
        symbols=clean["symbol"].nunique(),
        sessions=len(actual),
        bars=len(clean),
        first_session=first.strftime("%Y-%m-%d"),
        last_session=last.strftime("%Y-%m-%d"),
        # A user's manifest cannot itself establish independently licensed data
        # or confirm that all corporate actions/delistings were captured.
        license_independently_verified=False,
        corporate_actions_independently_verified=False,
        historical_universe_independently_verified=False,
        research_grade="UNVERIFIED_PROVENANCE_AND_ACTIONS",
        order_execution_permitted=False,
    )
