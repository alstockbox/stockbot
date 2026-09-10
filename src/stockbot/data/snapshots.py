from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd

from stockbot.data.market_schema import validate_canonical_bars
from stockbot.data.schemas import DataGrade

SCHEMA_VERSION = "1.2"
_SECRET_FRAGMENTS = ("token", "authorization", "api_key", "apikey", "secret", "password")


@dataclass(frozen=True)
class SnapshotManifestInput:
    provider: str
    grade: DataGrade
    symbols: tuple[str, ...]
    start: str
    end: str
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SnapshotManifest:
    snapshot_id: str
    provider: str
    grade: DataGrade
    symbols: tuple[str, ...]
    start: str
    end: str
    created_at: datetime
    row_count: int
    first_observation: dict[str, str]
    last_observation: dict[str, str]
    dataset_fingerprint: str
    schema_version: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class MarketSnapshot:
    snapshot_id: str
    path: Path
    bars: pd.DataFrame
    manifest: SnapshotManifest


def _safe_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(provenance)
    for key in clean:
        lowered = str(key).lower()
        if any(fragment in lowered for fragment in _SECRET_FRAGMENTS):
            raise ValueError("snapshot provenance cannot contain secret fields")
    try:
        json.dumps(clean, sort_keys=True)
    except TypeError as exc:
        raise ValueError("snapshot provenance must be JSON serializable") from exc
    return clean


def _normalized_symbols(symbols) -> tuple[str, ...]:
    result = tuple(sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}))
    if not result:
        raise ValueError("snapshot symbols cannot be empty")
    return result


def _stable_bar_frame(bars: pd.DataFrame) -> pd.DataFrame:
    validate_canonical_bars(bars)
    stable = bars.copy()
    stable["timestamp"] = pd.to_datetime(stable["timestamp"], utc=True)
    stable["symbol"] = stable["symbol"].astype(str).str.upper()
    stable["provider"] = stable["provider"].astype(str)
    numeric_columns = (
        "open", "high", "low", "close", "volume",
        "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume",
        "div_cash", "split_factor",
    )
    for column in numeric_columns:
        # Float64 + NaN is the canonical persisted representation. This avoids a
        # fingerprint changing merely because CSV reload converts None/object to NaN/float.
        stable[column] = pd.to_numeric(stable[column], errors="coerce").astype("float64")
    stable = stable.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
    # Retrieval time is lineage metadata, not market content, and therefore must not
    # make an otherwise identical market dataset receive a different fingerprint.
    stable = stable.drop(columns=["retrieved_at"])
    return stable


def compute_snapshot_fingerprint(bars: pd.DataFrame, manifest_input: SnapshotManifestInput) -> str:
    stable = _stable_bar_frame(bars)
    payload = {
        "provider": str(manifest_input.provider),
        "grade": manifest_input.grade.value,
        "symbols": _normalized_symbols(manifest_input.symbols),
        "start": pd.Timestamp(manifest_input.start).date().isoformat(),
        "end": pd.Timestamp(manifest_input.end).date().isoformat(),
        "schema_version": SCHEMA_VERSION,
        "provenance": _safe_provenance(manifest_input.provenance),
        "columns": [str(column) for column in stable.columns],
        "dtypes": [str(dtype) for dtype in stable.dtypes],
    }
    hasher = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    hashed = pd.util.hash_pandas_object(stable, index=False, categorize=False)
    hasher.update(hashed.to_numpy().tobytes())
    return hasher.hexdigest()


def _manifest_to_json(manifest: SnapshotManifest) -> str:
    payload = asdict(manifest)
    payload["grade"] = manifest.grade.value
    payload["symbols"] = list(manifest.symbols)
    payload["created_at"] = manifest.created_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return json.dumps(payload, sort_keys=True, indent=2)


def _manifest_from_dict(payload: dict[str, Any]) -> SnapshotManifest:
    created = datetime.fromisoformat(str(payload["created_at"]).replace("Z", "+00:00"))
    return SnapshotManifest(
        snapshot_id=str(payload["snapshot_id"]),
        provider=str(payload["provider"]),
        grade=DataGrade(payload["grade"]),
        symbols=tuple(payload["symbols"]),
        start=str(payload["start"]),
        end=str(payload["end"]),
        created_at=created,
        row_count=int(payload["row_count"]),
        first_observation={str(k): str(v) for k, v in payload["first_observation"].items()},
        last_observation={str(k): str(v) for k, v in payload["last_observation"].items()},
        dataset_fingerprint=str(payload["dataset_fingerprint"]),
        schema_version=str(payload["schema_version"]),
        provenance=dict(payload.get("provenance") or {}),
    )


class SnapshotStore:
    def __init__(self, root) -> None:
        self.root = Path(root)

    def write(
        self,
        bars: pd.DataFrame,
        manifest_input: SnapshotManifestInput,
        *,
        created_at: datetime | None = None,
    ) -> MarketSnapshot:
        validate_canonical_bars(bars)
        symbols = _normalized_symbols(manifest_input.symbols)
        actual_symbols = tuple(sorted(bars["symbol"].astype(str).str.upper().unique()))
        if actual_symbols != symbols:
            raise ValueError("snapshot bars do not match requested universe")
        provenance = _safe_provenance(manifest_input.provenance)
        created = created_at or datetime.now(timezone.utc)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        else:
            created = created.astimezone(timezone.utc)
        fingerprint = compute_snapshot_fingerprint(bars, manifest_input)
        provider_slug = re.sub(r"[^a-z0-9-]+", "-", manifest_input.provider.lower()).strip("-") or "provider"
        snapshot_id = f"{created.strftime('%Y%m%dT%H%M%SZ')}-{provider_slug}-{fingerprint[:8]}"
        path = self.root / snapshot_id
        if path.exists():
            raise FileExistsError(f"snapshot already exists: {snapshot_id}")

        canonical = bars.copy()
        canonical["timestamp"] = pd.to_datetime(canonical["timestamp"], utc=True)
        canonical["retrieved_at"] = pd.to_datetime(canonical["retrieved_at"], utc=True)
        canonical["symbol"] = canonical["symbol"].astype(str).str.upper()
        canonical = canonical.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
        grouped = canonical.groupby("symbol", sort=True)["timestamp"]
        first = {symbol: value.isoformat() for symbol, value in grouped.min().items()}
        last = {symbol: value.isoformat() for symbol, value in grouped.max().items()}
        manifest = SnapshotManifest(
            snapshot_id=snapshot_id,
            provider=manifest_input.provider,
            grade=manifest_input.grade,
            symbols=symbols,
            start=pd.Timestamp(manifest_input.start).date().isoformat(),
            end=pd.Timestamp(manifest_input.end).date().isoformat(),
            created_at=created,
            row_count=len(canonical),
            first_observation=first,
            last_observation=last,
            dataset_fingerprint=fingerprint,
            schema_version=SCHEMA_VERSION,
            provenance=provenance,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        path.mkdir(parents=False, exist_ok=False)
        try:
            canonical.to_csv(path / "bars.csv", index=False)
            (path / "manifest.json").write_text(_manifest_to_json(manifest), encoding="utf-8")
        except Exception:
            # Avoid leaving a half-written snapshot that looks valid.
            for child in path.iterdir():
                child.unlink(missing_ok=True)
            path.rmdir()
            raise
        return MarketSnapshot(snapshot_id, path, canonical, manifest)

    def load(self, snapshot_id: str) -> MarketSnapshot:
        snapshot_id = str(snapshot_id)
        path = self.root / snapshot_id
        manifest_path = path / "manifest.json"
        bars_path = path / "bars.csv"
        if not manifest_path.is_file() or not bars_path.is_file():
            raise FileNotFoundError(f"invalid snapshot: {snapshot_id}")
        manifest = _manifest_from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        if manifest.snapshot_id != snapshot_id:
            raise ValueError("snapshot manifest ID does not match directory")
        bars = pd.read_csv(bars_path)
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        bars["retrieved_at"] = pd.to_datetime(bars["retrieved_at"], utc=True)
        validate_canonical_bars(bars)
        if len(bars) != manifest.row_count:
            raise ValueError("snapshot row count does not match manifest")
        expected = SnapshotManifestInput(
            provider=manifest.provider,
            grade=manifest.grade,
            symbols=manifest.symbols,
            start=manifest.start,
            end=manifest.end,
            provenance=manifest.provenance,
        )
        if compute_snapshot_fingerprint(bars, expected) != manifest.dataset_fingerprint:
            raise ValueError("snapshot fingerprint mismatch")
        return MarketSnapshot(snapshot_id, path, bars, manifest)
