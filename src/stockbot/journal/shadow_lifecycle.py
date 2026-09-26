from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShadowLifecycleEvent:
    event_id: str
    decision_id: str
    event_type: str
    timestamp: datetime
    symbol: str
    price: float
    pnl_usd: float = 0.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id is required")
        if not self.decision_id.strip():
            raise ValueError("decision_id is required")
        if self.event_type not in {"OPEN", "CLOSE"}:
            raise ValueError("event_type must be OPEN or CLOSE")
        if not self.symbol.strip():
            raise ValueError("symbol is required")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")


@dataclass(frozen=True)
class ShadowLifecycleEntry:
    sequence: int
    previous_hash: str
    entry_hash: str
    event: ShadowLifecycleEvent


class AppendOnlyShadowLifecycleJournal:
    GENESIS_HASH = "0" * 64

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @staticmethod
    def _payload(event: ShadowLifecycleEvent) -> dict[str, Any]:
        payload = asdict(event)
        payload["timestamp"] = event.timestamp.astimezone(timezone.utc).isoformat()
        payload["metadata"] = dict(event.metadata or {})
        return payload

    @staticmethod
    def _hash(sequence: int, previous_hash: str, payload: dict[str, Any]) -> str:
        canonical = json.dumps(
            {
                "sequence": int(sequence),
                "previous_hash": previous_hash,
                "event": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def entries(self) -> list[ShadowLifecycleEntry]:
        if not self.path.exists():
            return []
        parsed: list[ShadowLifecycleEntry] = []
        previous = self.GENESIS_HASH
        expected_sequence = 1
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            if int(row["sequence"]) != expected_sequence:
                raise ValueError("shadow lifecycle sequence integrity failure")
            if row["previous_hash"] != previous:
                raise ValueError("shadow lifecycle hash-chain integrity failure")
            payload = dict(row["event"])
            expected_hash = self._hash(expected_sequence, previous, payload)
            if row["entry_hash"] != expected_hash:
                raise ValueError("shadow lifecycle entry integrity failure")
            event = ShadowLifecycleEvent(
                event_id=payload["event_id"],
                decision_id=payload["decision_id"],
                event_type=payload["event_type"],
                timestamp=datetime.fromisoformat(payload["timestamp"]),
                symbol=payload["symbol"],
                price=float(payload["price"]),
                pnl_usd=float(payload.get("pnl_usd", 0.0)),
                metadata=dict(payload.get("metadata", {})),
            )
            parsed.append(
                ShadowLifecycleEntry(
                    expected_sequence,
                    previous,
                    row["entry_hash"],
                    event,
                )
            )
            previous = row["entry_hash"]
            expected_sequence += 1
        return parsed

    def append(self, event: ShadowLifecycleEvent) -> ShadowLifecycleEntry:
        existing = self.entries()
        if any(row.event.event_id == event.event_id for row in existing):
            raise ValueError("event_id must be unique")
        sequence = len(existing) + 1
        previous = existing[-1].entry_hash if existing else self.GENESIS_HASH
        payload = self._payload(event)
        entry_hash = self._hash(sequence, previous, payload)
        row = {
            "sequence": sequence,
            "previous_hash": previous,
            "entry_hash": entry_hash,
            "event": payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )
        return ShadowLifecycleEntry(sequence, previous, entry_hash, event)

    def realized_pnl_for_date(self, day: date) -> float:
        return float(
            sum(
                row.event.pnl_usd
                for row in self.entries()
                if row.event.event_type == "CLOSE"
                and row.event.timestamp.date() == day
            )
        )
