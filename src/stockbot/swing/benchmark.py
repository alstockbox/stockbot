"""License-aware, offline total-return index comparison for XSTO swing OOS.

Only compares already computed OOS folds with user-supplied index levels.
Never fetches or redistributes index values; no broker connectivity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
from typing import Mapping

import exchange_calendars as xcals
import pandas as pd

from stockbot.swing.walkforward import WalkForwardResult


SUPPORTED_GI = {"OMXSBGI", "OMXS30GI", "OMXSGI"}


@dataclass(frozen=True)
class IndexProvenance:
    index_code: str
    source: str
    usage_rights_reference: str
    collected_at_utc: str
    currency: str = "SEK"
    return_basis: str = "GROSS_TOTAL_RETURN"

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "IndexProvenance":
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            # Consistent, explicit schema: callers must not mistake missing
            # provenance for verified licensing.
            raise ValueError("incomplete or unknown index provenance fields")
        return cls(**raw)

    def __post_init__(self) -> None:
        if self.index_code not in SUPPORTED_GI:
            raise ValueError("index must be a supported Swedish gross-total-return index")
        if self.currency != "SEK" or self.return_basis != "GROSS_TOTAL_RETURN":
            raise ValueError("SEK gross total return required")
        if (not isinstance(self.source, str) or not self.source.strip() or
            not isinstance(self.usage_rights_reference, str) or
            not self.usage_rights_reference.strip()):
            raise ValueError("index data source and rights reference required")
        _parse_utc(self.collected_at_utc)


def _parse_utc(raw: str) -> datetime:
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("invalid index collection timestamp") from exc
    if value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
        raise ValueError("index collection timestamp must use UTC")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class BenchmarkFold:
    test_start: str
    test_end: str
    strategy_return_pct: float
    benchmark_return_pct: float
    excess_return_pp: float


@dataclass(frozen=True)
class BenchmarkComparison:
    index_code: str
    mode: str
    benchmark_source_sha256: str
    benchmark_manifest_sha256: str
    license_independently_verified: bool
    external_index_history_verified: bool
    alpha_claim_permitted: bool
    folds: tuple[BenchmarkFold, ...]
    chained_benchmark_return_pct: float
    chained_strategy_return_pct: float
    apparent_excess_return_pp: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compare_oos_to_index(
    result: WalkForwardResult,
    levels: pd.DataFrame,
    provenance: IndexProvenance,
) -> BenchmarkComparison:
    """Require complete XSTO closes across EVERY fold, including prior close.

    Comparison is not tradable alpha: portfolio starts flat every fold,
    market costs differ, source rights unverified, and survivorship risk remains.
    """
    if result.mode != "WALK_FORWARD_BACKTEST_ONLY" or not result.folds:
        raise ValueError("benchmark requires walk-forward OOS folds")
    if not {"date", "close"}.issubset(levels.columns):
        raise ValueError("index levels require date,close")
    if len(levels) < 2:
        raise ValueError("insufficient index closes")
    bars = levels[["date", "close"]].copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="raise")
    if (bars["date"].isna().any() or
        (bars["date"] != bars["date"].dt.normalize()).any()):
        raise ValueError("daily index dates required")
    if bars["date"].duplicated().any():
        raise ValueError("duplicate benchmark date")
    bars["close"] = pd.to_numeric(bars["close"], errors="raise")
    if not bars["close"].map(isfinite).all() or (bars["close"] <= 0).any():
        raise ValueError("invalid benchmark close")
    bars = bars.sort_values("date").set_index("date")
    last_close = bars.index[-1].replace(hour=17, minute=30).tz_localize(
        "Europe/Stockholm"
    ).tz_convert("UTC").to_pydatetime()
    if _parse_utc(provenance.collected_at_utc) < last_close:
        raise ValueError("index data collected before last session close")
    cal = xcals.get_calendar("XSTO")
    expected_all = pd.DatetimeIndex(cal.sessions_in_range(
        bars.index[0], bars.index[-1]
    )).tz_localize(None).normalize()
    if len(bars.index.difference(expected_all)):
        raise ValueError("index contains non-XSTO date")
    all_returns: list[BenchmarkFold] = []
    chained_bench = 1.0
    chained_strategy = 1.0
    for fold in result.folds:
        anchor = pd.Timestamp(fold.train_end)
        end = pd.Timestamp(fold.test_end)
        sessions = pd.DatetimeIndex(cal.sessions_in_range(
            anchor, end
        )).tz_localize(None).normalize()
        if len(sessions) < 2 or not sessions.isin(bars.index).all():
            raise ValueError(f"missing XSTO index sessions for fold {fold.number}")
        start_level = float(bars.loc[anchor, "close"])
        end_level = float(bars.loc[end, "close"])
        bench_pct = (end_level / start_level - 1) * 100
        strategy_pct = float(fold.test_net_return_pct)
        chained_bench *= 1 + bench_pct / 100
        chained_strategy *= 1 + strategy_pct / 100
        all_returns.append(BenchmarkFold(
            fold.test_start, fold.test_end, strategy_pct,
            round(bench_pct, 6), round(strategy_pct - bench_pct, 6),
        ))
    canonical = bars.reset_index().to_csv(
        index=False, date_format="%Y-%m-%d", float_format="%.12g"
    ).encode()
    data_digest = sha256(canonical).hexdigest()
    manifest_digest = sha256(json.dumps(
        asdict(provenance), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False
    ).encode()).hexdigest()
    return BenchmarkComparison(
        index_code=provenance.index_code,
        mode="UNVERIFIED_OOS_BENCHMARK_COMPARISON",
        benchmark_source_sha256=data_digest,
        benchmark_manifest_sha256=manifest_digest,
        license_independently_verified=False,
        external_index_history_verified=False,
        alpha_claim_permitted=False,
        folds=tuple(all_returns),
        chained_benchmark_return_pct=round((chained_bench-1)*100, 6),
        chained_strategy_return_pct=round((chained_strategy-1)*100, 6),
        apparent_excess_return_pp=round((chained_strategy-chained_bench)*100, 6),
    )
