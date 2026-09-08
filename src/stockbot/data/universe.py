from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class UniverseMembership:
    symbol: str
    effective_from: str
    effective_to: str | None = None
    exchange: str | None = None
    sector: str | None = None
    industry: str | None = None
    delisted: bool = False


@dataclass(frozen=True)
class UniverseManifest:
    source: str
    survivorship_bias_controlled: bool
    includes_delisted_securities: bool
    point_in_time_membership: bool
    asof_semantics: str = "effective_interval"


@dataclass(frozen=True)
class UniverseCoverageReport:
    total_bar_rows: int
    covered_bar_rows: int
    membership_coverage: float
    symbols_in_bars: int
    symbols_with_membership: int
    missing_symbols: tuple[str, ...]
    survivorship_bias_controlled: bool
    includes_delisted_securities: bool
    point_in_time_membership: bool
    research_grade_universe: bool
    reasons: tuple[str, ...]


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


class PointInTimeUniverse:
    """Historical membership intervals used to avoid today's-universe backtests."""

    def __init__(
        self,
        memberships: list[UniverseMembership] | tuple[UniverseMembership, ...],
        manifest: UniverseManifest,
    ) -> None:
        if not memberships:
            raise ValueError("point-in-time universe requires membership history")
        self.memberships = tuple(memberships)
        self.manifest = manifest
        self._validate()

    def _validate(self) -> None:
        by_symbol: dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None]]] = {}
        for row in self.memberships:
            symbol = row.symbol.strip().upper()
            if not symbol:
                raise ValueError("universe membership symbol cannot be empty")
            start = _utc(row.effective_from)
            end = None if row.effective_to is None else _utc(row.effective_to)
            if end is not None and end <= start:
                raise ValueError("universe effective_to must be after effective_from")
            by_symbol.setdefault(symbol, []).append((start, end))

        for symbol, intervals in by_symbol.items():
            ordered = sorted(intervals, key=lambda item: item[0])
            for previous, current in zip(ordered, ordered[1:]):
                previous_end = previous[1]
                if previous_end is None or current[0] < previous_end:
                    raise ValueError(f"overlapping universe membership intervals for {symbol}")

    def members_at(self, timestamp: str | pd.Timestamp) -> tuple[str, ...]:
        asof = _utc(timestamp)
        members: set[str] = set()
        for row in self.memberships:
            start = _utc(row.effective_from)
            end = None if row.effective_to is None else _utc(row.effective_to)
            if start <= asof and (end is None or asof < end):
                members.add(row.symbol.strip().upper())
        return tuple(sorted(members))

    def metadata_at(self, symbol: str, timestamp: str | pd.Timestamp) -> UniverseMembership | None:
        target = symbol.strip().upper()
        asof = _utc(timestamp)
        for row in self.memberships:
            if row.symbol.strip().upper() != target:
                continue
            start = _utc(row.effective_from)
            end = None if row.effective_to is None else _utc(row.effective_to)
            if start <= asof and (end is None or asof < end):
                return row
        return None

    def coverage_for_bars(
        self,
        bars: pd.DataFrame,
        *,
        min_membership_coverage: float = 0.99,
    ) -> UniverseCoverageReport:
        if not 0.0 < min_membership_coverage <= 1.0:
            raise ValueError("min_membership_coverage must be in (0,1]")
        if not {"symbol", "timestamp"}.issubset(bars.columns):
            raise ValueError("bars require symbol and timestamp columns")

        covered = 0
        symbols = {str(value).strip().upper() for value in bars["symbol"]}
        symbols_with_membership: set[str] = set()
        for symbol, timestamp in zip(bars["symbol"], bars["timestamp"]):
            normalized = str(symbol).strip().upper()
            if self.metadata_at(normalized, timestamp) is not None:
                covered += 1
                symbols_with_membership.add(normalized)

        total = len(bars)
        coverage = covered / total if total else 0.0
        missing_symbols = tuple(sorted(symbols.difference(symbols_with_membership)))
        reasons: list[str] = []
        if coverage < min_membership_coverage:
            reasons.append("insufficient_point_in_time_membership_coverage")
        if not self.manifest.point_in_time_membership:
            reasons.append("membership_not_point_in_time")
        if not self.manifest.survivorship_bias_controlled:
            reasons.append("survivorship_bias_not_controlled")
        if not self.manifest.includes_delisted_securities:
            reasons.append("delisted_securities_not_included")

        return UniverseCoverageReport(
            total_bar_rows=total,
            covered_bar_rows=covered,
            membership_coverage=float(coverage),
            symbols_in_bars=len(symbols),
            symbols_with_membership=len(symbols_with_membership),
            missing_symbols=missing_symbols,
            survivorship_bias_controlled=self.manifest.survivorship_bias_controlled,
            includes_delisted_securities=self.manifest.includes_delisted_securities,
            point_in_time_membership=self.manifest.point_in_time_membership,
            research_grade_universe=not reasons,
            reasons=tuple(reasons),
        )
