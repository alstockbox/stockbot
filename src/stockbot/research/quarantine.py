from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class QuarantineConfig:
    start: str
    min_development_periods: int = 252
    min_quarantine_periods: int = 42

    def __post_init__(self) -> None:
        try:
            pd.Timestamp(self.start)
        except Exception as exc:
            raise ValueError("quarantine start must be a valid timestamp/date") from exc
        if self.min_development_periods <= 0:
            raise ValueError("min_development_periods must be positive")
        if self.min_quarantine_periods <= 0:
            raise ValueError("min_quarantine_periods must be positive")


@dataclass(frozen=True)
class QuarantineManifest:
    quarantine_id: str
    start: str
    development_periods: int
    quarantine_periods: int
    development_rows: int
    quarantine_rows: int
    created_at: str
    state: str = "sealed"
    schema_version: int = 1


@dataclass(frozen=True)
class QuarantineSplit:
    development_bars: pd.DataFrame
    quarantine_bars: pd.DataFrame
    manifest: QuarantineManifest


def _utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _sealed_content_fingerprint(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    canonical["timestamp"] = pd.to_datetime(canonical["timestamp"], utc=True)
    canonical["symbol"] = canonical["symbol"].astype(str).str.upper()
    columns = tuple(sorted(canonical.columns, key=str))
    canonical = canonical.loc[:, list(columns)]
    row_hashes = sorted(
        int(value)
        for value in pd.util.hash_pandas_object(
            canonical,
            index=False,
            categorize=False,
        ).tolist()
    )
    payload = {
        "columns": [str(value) for value in columns],
        "dtypes": [str(canonical[column].dtype) for column in columns],
        "row_hashes": row_hashes,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def split_sealed_quarantine(
    bars: pd.DataFrame,
    config: QuarantineConfig,
) -> QuarantineSplit:
    """Split a dataset at an explicit, immutable research-cycle boundary.

    Routine research must receive only ``development_bars``. Every observation at or
    after the sealed start belongs to quarantine and must not be used for model search,
    policy search, feature selection, stacking, diagnostics or ordinary holdout tuning.

    The boundary is explicit rather than "last N percent" so future nightly snapshots
    cannot silently move the audit set and leak prior quarantine observations back into
    development. Rotating the boundary is a separate, explicit research-cycle action.
    """

    required = {"timestamp", "symbol"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")

    frame = bars.copy()
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    start = _utc_timestamp(config.start)
    development = frame.loc[timestamps < start].copy()
    quarantine = frame.loc[timestamps >= start].copy()

    development_dates = pd.DatetimeIndex(pd.to_datetime(development["timestamp"], utc=True).unique()).sort_values()
    quarantine_dates = pd.DatetimeIndex(pd.to_datetime(quarantine["timestamp"], utc=True).unique()).sort_values()
    if len(development_dates) < config.min_development_periods:
        raise ValueError(
            f"sealed quarantine leaves only {len(development_dates)} development periods; "
            f"need at least {config.min_development_periods}"
        )
    if len(quarantine_dates) < config.min_quarantine_periods:
        raise ValueError(
            f"sealed quarantine contains only {len(quarantine_dates)} periods; "
            f"need at least {config.min_quarantine_periods}"
        )

    payload = {
        "start": start.isoformat(),
        "development_periods": len(development_dates),
        "quarantine_periods": len(quarantine_dates),
        "development_rows": len(development),
        "quarantine_rows": len(quarantine),
        "quarantine_content_fingerprint": _sealed_content_fingerprint(quarantine),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    quarantine_id = hashlib.sha256(raw).hexdigest()[:24]
    manifest = QuarantineManifest(
        quarantine_id=quarantine_id,
        start=start.isoformat(),
        development_periods=len(development_dates),
        quarantine_periods=len(quarantine_dates),
        development_rows=len(development),
        quarantine_rows=len(quarantine),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return QuarantineSplit(
        development_bars=development,
        quarantine_bars=quarantine,
        manifest=manifest,
    )


def load_quarantine_manifest(path: str | Path) -> QuarantineManifest | None:
    target = Path(path)
    if not target.exists():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    manifest = QuarantineManifest(**payload)
    if manifest.state != "sealed":
        raise ValueError("quarantine manifest is not in sealed state")
    return manifest


def write_quarantine_manifest(path: str | Path, manifest: QuarantineManifest) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(manifest), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def assert_same_quarantine_boundary(
    manifest: QuarantineManifest,
    config: QuarantineConfig,
) -> None:
    """Prevent routine jobs from silently changing a previously sealed boundary."""

    expected = _utc_timestamp(manifest.start)
    requested = _utc_timestamp(config.start)
    if expected != requested:
        raise ValueError(
            "quarantine boundary is sealed; rotating it requires an explicit new research cycle"
        )


def persist_sealed_quarantine(
    path: str | Path,
    split: QuarantineSplit,
    config: QuarantineConfig,
) -> QuarantineManifest:
    """Persist a quarantine manifest while refusing silent boundary rotation.

    Counts and the content-derived quarantine id may evolve as new future observations
    arrive, but the start boundary cannot move. This lets a quarantine grow without
    allowing previous audit observations back into routine research.
    """

    existing = load_quarantine_manifest(path)
    if existing is not None:
        assert_same_quarantine_boundary(existing, config)
        created_at = existing.created_at
        manifest = QuarantineManifest(
            quarantine_id=split.manifest.quarantine_id,
            start=existing.start,
            development_periods=split.manifest.development_periods,
            quarantine_periods=split.manifest.quarantine_periods,
            development_rows=split.manifest.development_rows,
            quarantine_rows=split.manifest.quarantine_rows,
            created_at=created_at,
        )
    else:
        manifest = split.manifest
    write_quarantine_manifest(path, manifest)
    return manifest
