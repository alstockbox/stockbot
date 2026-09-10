from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from stockbot.paper.locking import (
    exclusive_paper_ledger_write_lock,
    paper_ledger_write_lock_path,
    shared_paper_ledger_read_lock,
)


_LEDGER_SCHEMA_VERSION = 2
_LEDGER_LEGACY_SCHEMA_VERSION = 1
_LEDGER_SUPPORTED_SCHEMA_VERSIONS = {_LEDGER_LEGACY_SCHEMA_VERSION, _LEDGER_SCHEMA_VERSION}
_LEDGER_SCHEMA_KEY = "__ledger_schema_version"
_LEDGER_PREVIOUS_HASH_KEY = "__ledger_previous_hash"
_LEDGER_RECORD_HASH_KEY = "__ledger_record_hash"
_LEDGER_METADATA_KEYS = {
    _LEDGER_SCHEMA_KEY,
    _LEDGER_PREVIOUS_HASH_KEY,
    _LEDGER_RECORD_HASH_KEY,
}
_LEDGER_GENESIS_HASH = hashlib.sha256(b"stockbot-paper-ledger-v1").hexdigest()
_HEAD_SCHEMA_VERSION = 1


def _parse_aware_timestamp(value: str, *, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed


@dataclass(frozen=True)
class PaperObservation:
    strategy_id: str
    timestamp: str
    net_return: float
    benchmark_return: float = 0.0
    turnover: float = 0.0
    cost_rate: float = 0.0
    fill_rate: float = 1.0
    signal_count: int = 0
    regime: str | None = None
    notes: str | None = None
    research_cycle_id: str | None = None
    model_artifact_id: str | None = None
    signal_timestamp: str | None = None
    signal_snapshot_fingerprint: str | None = None
    realization_snapshot_fingerprint: str | None = None
    gross_return: float | None = None
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id is required")
        realization_time = _parse_aware_timestamp(self.timestamp, name="timestamp")
        if self.signal_timestamp is not None:
            signal_time = _parse_aware_timestamp(self.signal_timestamp, name="signal_timestamp")
            if signal_time >= realization_time:
                raise ValueError("signal_timestamp must precede realization timestamp")
        for name, value in (
            ("net_return", self.net_return),
            ("benchmark_return", self.benchmark_return),
            ("turnover", self.turnover),
            ("cost_rate", self.cost_rate),
            ("fill_rate", self.fill_rate),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.gross_return is not None and not math.isfinite(float(self.gross_return)):
            raise ValueError("gross_return must be finite")
        if self.turnover < 0.0 or self.cost_rate < 0.0:
            raise ValueError("turnover and cost_rate must be non-negative")
        if not 0.0 <= self.fill_rate <= 1.0:
            raise ValueError("fill_rate must be in [0,1]")
        if self.signal_count < 0:
            raise ValueError("signal_count must be non-negative")


def _canonical_observation_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _next_ledger_hash(previous_hash: str, observation_payload: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(previous_hash).encode("ascii"))
    hasher.update(b"\n")
    hasher.update(_canonical_observation_payload(observation_payload))
    return hasher.hexdigest()


def _split_ledger_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    present = _LEDGER_METADATA_KEYS.intersection(payload)
    if not present:
        return dict(payload), None
    if present != _LEDGER_METADATA_KEYS:
        raise ValueError("paper ledger integrity metadata is incomplete")
    observation_payload = {
        key: value for key, value in payload.items() if key not in _LEDGER_METADATA_KEYS
    }
    metadata = {
        _LEDGER_SCHEMA_KEY: payload[_LEDGER_SCHEMA_KEY],
        _LEDGER_PREVIOUS_HASH_KEY: payload[_LEDGER_PREVIOUS_HASH_KEY],
        _LEDGER_RECORD_HASH_KEY: payload[_LEDGER_RECORD_HASH_KEY],
    }
    return observation_payload, metadata


class PaperTradingLedger:
    """Append-only, hash-chained JSONL ledger for forward paper observations.

    Legacy unchained rows and v1 chained rows remain readable for backward compatibility.
    New v2 rows require an atomically replaced head checkpoint that binds the committed
    tail hash, logical row count and ledger byte size. This closes the ordinary hash-chain
    tail-truncation gap: removing the latest committed row(s), deleting the checkpoint, or
    appending data without committing a matching checkpoint fails closed on verification.

    Verify + chronology check + duplicate-check + append + checkpoint commit is serialized
    by a per-ledger OS writer lock. Public verified reads wait on a shared lock so they see
    either the previous committed state or the fully committed ledger+checkpoint pair.
    Each strategy's realization timestamps must advance strictly forward.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def head_path(self) -> Path:
        return Path(str(self.path) + ".head.json")

    def _verify_head_checkpoint(
        self,
        *,
        tail_hash: str,
        row_count: int,
        checkpoint_required: bool,
    ) -> None:
        if not self.head_path.exists():
            if checkpoint_required:
                raise ValueError("paper ledger checkpoint is missing for schema-v2 evidence")
            return
        if not self.head_path.is_file():
            raise ValueError("paper ledger checkpoint is not a regular file")
        try:
            payload = json.loads(self.head_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("paper ledger checkpoint integrity violation: invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("paper ledger checkpoint integrity violation: checkpoint must be an object")
        try:
            schema_version = int(payload["schema_version"])
            checkpoint_rows = int(payload["row_count"])
            checkpoint_size = int(payload["ledger_size_bytes"])
            checkpoint_tail = str(payload["tail_hash"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("paper ledger checkpoint integrity violation: invalid fields") from exc
        if schema_version != _HEAD_SCHEMA_VERSION:
            raise ValueError("paper ledger checkpoint integrity violation: unsupported schema")
        if checkpoint_rows != int(row_count):
            raise ValueError("paper ledger checkpoint integrity violation: row count mismatch")
        if checkpoint_tail != str(tail_hash):
            raise ValueError("paper ledger checkpoint integrity violation: tail hash mismatch")
        try:
            current_size = self.path.stat().st_size
        except OSError as exc:
            raise ValueError("paper ledger checkpoint integrity violation: ledger stat failed") from exc
        if checkpoint_size != current_size:
            raise ValueError("paper ledger checkpoint integrity violation: ledger size mismatch")

    def _write_head_checkpoint(self, *, tail_hash: str, row_count: int) -> None:
        payload = {
            "schema_version": _HEAD_SCHEMA_VERSION,
            "row_count": int(row_count),
            "tail_hash": str(tail_hash),
            "ledger_size_bytes": int(self.path.stat().st_size),
        }
        serialized = json.dumps(payload, sort_keys=True, indent=2) + "\n"
        temporary = Path(str(self.head_path) + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.head_path)
            directory_fd = os.open(str(self.head_path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _read_verified(self) -> tuple[list[PaperObservation], str]:
        if not self.path.exists():
            if self.head_path.exists():
                raise ValueError("paper ledger checkpoint exists without ledger")
            return [], _LEDGER_GENESIS_HASH

        rows: list[PaperObservation] = []
        tail_hash = _LEDGER_GENESIS_HASH
        chain_started = False
        checkpoint_required = False
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw_payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"paper ledger integrity violation at line {line_number}: invalid JSON"
                ) from exc
            if not isinstance(raw_payload, dict):
                raise ValueError(
                    f"paper ledger integrity violation at line {line_number}: row must be an object"
                )

            observation_payload, metadata = _split_ledger_payload(raw_payload)
            try:
                observation = PaperObservation(**observation_payload)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"paper ledger integrity violation at line {line_number}: invalid observation"
                ) from exc

            expected_hash = _next_ledger_hash(tail_hash, observation_payload)
            if metadata is None:
                if chain_started:
                    raise ValueError(
                        f"paper ledger integrity violation at line {line_number}: unchained row after chain start"
                    )
                tail_hash = expected_hash
            else:
                try:
                    schema_version = int(metadata[_LEDGER_SCHEMA_KEY])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"paper ledger integrity violation at line {line_number}: invalid chain schema"
                    ) from exc
                if schema_version not in _LEDGER_SUPPORTED_SCHEMA_VERSIONS:
                    raise ValueError(
                        f"paper ledger integrity violation at line {line_number}: unsupported chain schema"
                    )
                if str(metadata[_LEDGER_PREVIOUS_HASH_KEY]) != tail_hash:
                    raise ValueError(
                        f"paper ledger integrity violation at line {line_number}: previous hash mismatch"
                    )
                if str(metadata[_LEDGER_RECORD_HASH_KEY]) != expected_hash:
                    raise ValueError(
                        f"paper ledger integrity violation at line {line_number}: record hash mismatch"
                    )
                tail_hash = expected_hash
                chain_started = True
                if schema_version >= _LEDGER_SCHEMA_VERSION:
                    checkpoint_required = True
            rows.append(observation)

        self._verify_head_checkpoint(
            tail_hash=tail_hash,
            row_count=len(rows),
            checkpoint_required=checkpoint_required,
        )
        return rows, tail_hash

    def append(self, observation: PaperObservation) -> None:
        lock_path = paper_ledger_write_lock_path(self.path)
        with exclusive_paper_ledger_write_lock(lock_path):
            existing, tail_hash = self._read_verified()
            key = (observation.strategy_id, observation.timestamp)
            if any((row.strategy_id, row.timestamp) == key for row in existing):
                raise ValueError("duplicate paper observation for strategy/timestamp")

            strategy_rows = [row for row in existing if row.strategy_id == observation.strategy_id]
            if strategy_rows:
                latest_existing = max(
                    _parse_aware_timestamp(row.timestamp, name="timestamp")
                    for row in strategy_rows
                )
                incoming = _parse_aware_timestamp(observation.timestamp, name="timestamp")
                if incoming <= latest_existing:
                    raise ValueError(
                        "paper observation timestamps must be strictly increasing per strategy"
                    )

            observation_payload = asdict(observation)
            record_hash = _next_ledger_hash(tail_hash, observation_payload)
            payload = dict(observation_payload)
            payload[_LEDGER_SCHEMA_KEY] = _LEDGER_SCHEMA_VERSION
            payload[_LEDGER_PREVIOUS_HASH_KEY] = tail_hash
            payload[_LEDGER_RECORD_HASH_KEY] = record_hash
            serialized = json.dumps(payload, sort_keys=True) + "\n"
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())

            self._write_head_checkpoint(
                tail_hash=record_hash,
                row_count=len(existing) + 1,
            )

    def records(self, *, strategy_id: str | None = None) -> list[PaperObservation]:
        lock_path = paper_ledger_write_lock_path(self.path)
        with shared_paper_ledger_read_lock(lock_path):
            rows, _ = self._read_verified()
        if strategy_id is not None:
            rows = [row for row in rows if row.strategy_id == strategy_id]
        rows.sort(key=lambda row: row.timestamp)
        return rows

    def strategy_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row.strategy_id for row in self.records()}))


def make_paper_observation(
    *,
    strategy_id: str,
    net_return: float,
    benchmark_return: float = 0.0,
    turnover: float = 0.0,
    cost_rate: float = 0.0,
    fill_rate: float = 1.0,
    signal_count: int = 0,
    regime: str | None = None,
    timestamp: str | None = None,
    notes: str | None = None,
    research_cycle_id: str | None = None,
    model_artifact_id: str | None = None,
    signal_timestamp: str | None = None,
    signal_snapshot_fingerprint: str | None = None,
    realization_snapshot_fingerprint: str | None = None,
    gross_return: float | None = None,
) -> PaperObservation:
    return PaperObservation(
        strategy_id=strategy_id,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        net_return=float(net_return),
        benchmark_return=float(benchmark_return),
        turnover=float(turnover),
        cost_rate=float(cost_rate),
        fill_rate=float(fill_rate),
        signal_count=int(signal_count),
        regime=regime,
        notes=notes,
        research_cycle_id=research_cycle_id,
        model_artifact_id=model_artifact_id,
        signal_timestamp=signal_timestamp,
        signal_snapshot_fingerprint=signal_snapshot_fingerprint,
        realization_snapshot_fingerprint=realization_snapshot_fingerprint,
        gross_return=(None if gross_return is None else float(gross_return)),
    )
