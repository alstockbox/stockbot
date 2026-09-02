from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from stockbot.data.market_schema import validate_canonical_bars
from stockbot.data.providers.http import ProviderError
from stockbot.data.schemas import DataGrade
from stockbot.data.snapshots import MarketSnapshot, SnapshotManifestInput, SnapshotStore


class DownloadError(RuntimeError):
    pass


def _symbols(values: Iterable[str]) -> tuple[str, ...]:
    result = tuple(sorted({str(value).strip().upper() for value in values if str(value).strip()}))
    if not result:
        raise ValueError("symbols cannot be empty")
    return result


def download_market_snapshot(
    provider,
    symbols: Iterable[str],
    start,
    end,
    store: SnapshotStore,
    *,
    grade: DataGrade | None = None,
    provenance: Mapping | None = None,
) -> MarketSnapshot:
    universe = _symbols(symbols)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError("start must be on or before end")
    provider_grade = getattr(provider, "grade", DataGrade.BOOTSTRAP)
    effective_grade = grade or provider_grade
    provenance = dict(provenance or {})
    provider_name = str(getattr(provider, "name", provider.__class__.__name__)).lower()

    if effective_grade is DataGrade.RESEARCH_GRADE and provider_grade is not DataGrade.RESEARCH_GRADE:
        tiingo_override = provider_name == "tiingo" and provenance.get("point_in_time_universe") is True
        if not tiingo_override:
            raise DownloadError("provider/data provenance is not eligible for research-grade promotion")
    if provider_name == "yahoo-bootstrap" and effective_grade is not DataGrade.BOOTSTRAP:
        raise DownloadError("Yahoo bootstrap data can only be BOOTSTRAP grade")

    frames = []
    try:
        for symbol in universe:
            frame = provider.fetch_bars(symbol, start_ts, end_ts)
            if frame is None or frame.empty:
                raise DownloadError(f"provider returned no data for {symbol}")
            validate_canonical_bars(frame)
            actual = set(frame["symbol"].astype(str).str.upper().unique())
            if actual != {symbol}:
                raise DownloadError(f"provider returned wrong symbol set for {symbol}")
            frames.append(frame)
    except (ProviderError, ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, DownloadError):
            raise
        raise DownloadError(f"market download failed before snapshot persistence: {type(exc).__name__}") from exc

    bars = pd.concat(frames, ignore_index=True)
    bars["symbol"] = bars["symbol"].astype(str).str.upper()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
    validate_canonical_bars(bars)
    if set(bars["symbol"].unique()) != set(universe):
        raise DownloadError("downloaded snapshot does not contain the complete requested universe")

    manifest_input = SnapshotManifestInput(
        provider=provider_name,
        grade=effective_grade,
        symbols=universe,
        start=start_ts.date().isoformat(),
        end=end_ts.date().isoformat(),
        provenance=provenance,
    )
    return store.write(bars, manifest_input)
