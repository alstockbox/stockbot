"""Single-writer, replay-verifiable SEK paper ledger (NO broker connectivity).

This module records *hypothetical* fills supplied by a research simulator.
USD-denominated shadow trading remains a different ledger/accounting context.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import re
from typing import Literal, Mapping

Side = Literal["BUY", "SELL"]
GENESIS = "0" * 64


@dataclass(frozen=True)
class PaperTrade:
    decision_id: str
    timestamp: datetime
    symbol: str
    side: Side
    quantity: int
    price_sek: float
    fee_sek: float

    def __post_init__(self) -> None:
        if not self.decision_id.strip():
            raise ValueError("decision ID is required")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.symbol):
            raise ValueError("invalid symbol")
        if self.side not in ("BUY", "SELL"):
            raise ValueError("invalid side")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) or self.quantity < 1:
            raise ValueError("quantity must be a positive whole number")
        if not isfinite(self.price_sek) or self.price_sek <= 0:
            raise ValueError("price_sek must be positive")
        if not isfinite(self.fee_sek) or self.fee_sek < 0:
            raise ValueError("fee_sek must be nonnegative")


@dataclass(frozen=True)
class PaperBalance:
    cash_sek: float
    positions: dict[str, int]
    last_sequence: int
    equity_sek: float | None


def _payload(event: PaperTrade) -> dict[str, object]:
    return {
        "decision_id": event.decision_id,
        "timestamp": event.timestamp.astimezone(timezone.utc).isoformat(),
        "symbol": event.symbol,
        "side": event.side,
        "quantity": event.quantity,
        "price_sek": event.price_sek,
        "fee_sek": event.fee_sek,
    }


def _hash(seq: int, previous: str, payload: dict[str, object], initial_cash_sek: float) -> str:
    raw = json.dumps({"sequence": seq, "previous_hash": previous, "event": payload, "initial_cash_sek": initial_cash_sek},
                     separators=(",", ":"), sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class SEKPaperLedger:
    """Strict, broker-free execution accounting with replay and integrity checks."""

    def __init__(self, path: str | Path, *, initial_cash_sek: float = 1_000.0) -> None:
        if not isfinite(initial_cash_sek) or initial_cash_sek <= 0:
            raise ValueError("initial_cash_sek must be positive")
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.initial_cash_sek = float(initial_cash_sek)

    def _replay(self) -> tuple[float, dict[str, int], list[dict[str, object]]]:
        cash = self.initial_cash_sek
        positions: dict[str, int] = {}
        records: list[dict[str, object]] = []
        previous = GENESIS
        ids: set[str] = set()
        if not self.path.exists():
            return cash, positions, records
        try:
            rows = self.path.read_text(encoding="utf-8").splitlines()
            for sequence, line in enumerate(rows, 1):
                row = json.loads(line)
                if row["sequence"] != sequence or row["previous_hash"] != previous or row.get("initial_cash_sek") != self.initial_cash_sek:
                    raise ValueError("paper ledger sequence/hash link mismatch")
                data = row["event"]
                if row["entry_hash"] != _hash(sequence, previous, data, self.initial_cash_sek):
                    raise ValueError("paper ledger hash mismatch")
                event = PaperTrade(
                    decision_id=data["decision_id"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    symbol=data["symbol"], side=data["side"],
                    quantity=data["quantity"], price_sek=data["price_sek"],
                    fee_sek=data["fee_sek"],
                )
                if event.decision_id in ids:
                    raise ValueError("duplicate decision id in paper ledger")
                ids.add(event.decision_id)
                amount = event.quantity * event.price_sek
                if event.side == "BUY":
                    total = amount + event.fee_sek
                    if total > cash + 1e-8:
                        raise ValueError("paper ledger overspent")
                    cash -= total
                    positions[event.symbol] = positions.get(event.symbol, 0) + event.quantity
                else:
                    held = positions.get(event.symbol, 0)
                    if event.quantity > held:
                        raise ValueError("paper ledger naked sale")
                    cash += amount - event.fee_sek
                    positions[event.symbol] = held - event.quantity
                    if positions[event.symbol] == 0:
                        del positions[event.symbol]
                if cash < -1e-8:
                    raise ValueError("paper ledger negative cash")
                previous = row["entry_hash"]
                records.append(row)
        except (OSError, UnicodeError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid paper ledger; fail closed") from exc
        return cash, positions, records

    def record_paper_fill(self, event: PaperTrade) -> PaperBalance:
        """Append a PAPER fill, never place a market order. Idempotency is fail-closed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                cash, positions, records = self._replay()
                if any(row["event"]["decision_id"] == event.decision_id for row in records):
                    raise ValueError("DUPLICATE_DECISION_ID")
                notional = event.price_sek * event.quantity
                if event.side == "BUY" and notional + event.fee_sek > cash + 1e-8:
                    raise ValueError("INSUFFICIENT_PAPER_CASH")
                if event.side == "SELL" and event.quantity > positions.get(event.symbol, 0):
                    raise ValueError("NAKED_SELL_BLOCKED")
                if event.side == "SELL" and notional < event.fee_sek:
                    raise ValueError("FEE_EXCEEDS_PROCEEDS")
                sequence = len(records) + 1
                prior = records[-1]["entry_hash"] if records else GENESIS
                data = _payload(event)
                record = {"sequence": sequence, "previous_hash": prior,
                          "initial_cash_sek": self.initial_cash_sek,
                          "entry_hash": _hash(sequence, prior, data, self.initial_cash_sek), "event": data}
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                return self._snapshot_unlocked()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _snapshot_unlocked(self, marks: Mapping[str, float] | None = None) -> PaperBalance:
        cash, positions, rows = self._replay()
        equity = None
        if not positions:
            equity = cash
        elif marks is not None:
            if not all(symbol in marks and isfinite(float(marks[symbol]))
                       and float(marks[symbol]) > 0 for symbol in positions):
                raise ValueError("complete valid marks required")
            equity = cash + sum(qty * float(marks[symbol])
                                for symbol, qty in positions.items())
        return PaperBalance(cash, positions, len(rows), equity)

    def find_decision(self, decision_id: str) -> PaperTrade | None:
        """Verify complete ledger before returning a recorded synthetic fill."""
        if not isinstance(decision_id, str) or not decision_id.strip():
            raise ValueError("decision ID required")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            try:
                _, _, rows = self._replay()
                for row in rows:
                    data = row["event"]
                    if data["decision_id"] == decision_id:
                        return PaperTrade(
                            decision_id=data["decision_id"],
                            timestamp=datetime.fromisoformat(data["timestamp"]),
                            symbol=data["symbol"], side=data["side"],
                            quantity=data["quantity"], price_sek=data["price_sek"],
                            fee_sek=data["fee_sek"]
                        )
                return None
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def has_decision(self, decision_id: str) -> bool:
        return self.find_decision(decision_id) is not None

    def snapshot(self, marks: Mapping[str, float] | None = None) -> PaperBalance:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            try:
                return self._snapshot_unlocked(marks)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
