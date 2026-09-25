from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DecisionEvent:
    decision_id: str
    timestamp: datetime
    symbol: str
    strategy: str
    regime: str
    signal_score: float
    confidence: float
    expected_return: float
    expected_cost: float
    expected_risk: float
    approved: bool
    reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.decision_id.strip():
            raise ValueError("decision_id is required")
        if not self.symbol.strip():
            raise ValueError("symbol is required")
        if not self.strategy.strip():
            raise ValueError("strategy is required")
        if not 0.0 <= float(self.signal_score) <= 1.0:
            raise ValueError("signal_score must be in [0,1]")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")


@dataclass(frozen=True)
class JournalEntry:
    sequence: int
    previous_hash: str
    entry_hash: str
    event: DecisionEvent


class AppendOnlyTradeJournal:
    GENESIS_HASH = "0" * 64

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @staticmethod
    def _event_payload(event: DecisionEvent) -> dict[str, Any]:
        payload = asdict(event)
        payload["timestamp"] = event.timestamp.astimezone(timezone.utc).isoformat()
        payload["reasons"] = list(event.reasons)
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

    def entries(self) -> list[JournalEntry]:
        if not self.path.exists():
            return []
        parsed: list[JournalEntry] = []
        previous = self.GENESIS_HASH
        expected_sequence = 1
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            if int(row["sequence"]) != expected_sequence:
                raise ValueError("journal sequence integrity failure")
            if row["previous_hash"] != previous:
                raise ValueError("journal hash-chain integrity failure")
            payload = dict(row["event"])
            expected_hash = self._hash(expected_sequence, previous, payload)
            if row["entry_hash"] != expected_hash:
                raise ValueError("journal entry integrity failure")
            event = DecisionEvent(
                decision_id=payload["decision_id"],
                timestamp=datetime.fromisoformat(payload["timestamp"]),
                symbol=payload["symbol"],
                strategy=payload["strategy"],
                regime=payload["regime"],
                signal_score=float(payload["signal_score"]),
                confidence=float(payload["confidence"]),
                expected_return=float(payload["expected_return"]),
                expected_cost=float(payload["expected_cost"]),
                expected_risk=float(payload["expected_risk"]),
                approved=bool(payload["approved"]),
                reasons=tuple(payload.get("reasons", [])),
                metadata=dict(payload.get("metadata", {})),
            )
            parsed.append(JournalEntry(expected_sequence, previous, row["entry_hash"], event))
            previous = row["entry_hash"]
            expected_sequence += 1
        return parsed

    def append(self, event: DecisionEvent) -> JournalEntry:
        existing = self.entries()
        if any(row.event.decision_id == event.decision_id for row in existing):
            raise ValueError("decision_id must be unique")
        sequence = len(existing) + 1
        previous = existing[-1].entry_hash if existing else self.GENESIS_HASH
        payload = self._event_payload(event)
        entry_hash = self._hash(sequence, previous, payload)
        row = {
            "sequence": sequence,
            "previous_hash": previous,
            "entry_hash": entry_hash,
            "event": payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        return JournalEntry(sequence, previous, entry_hash, event)
