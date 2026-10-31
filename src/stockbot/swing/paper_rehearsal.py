"""Broker-free two-phase synthetic EOD -> next-XSTO-open rehearsal.

Only accepts the SYNTHETIC_TEST_ONLY feed. A decision must be journaled
BEFORE the target session opens; target OPEN is reconstructed AFTER its
completed daily candle arrives. This is NOT actual prospective paper trading.
The separate SEKPaperLedger is the accounting source of truth on recovery.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import fcntl
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
from typing import Literal
from uuid import uuid4

import exchange_calendars as xcals
import pandas as pd

from stockbot.swing.paper_feed import PaperFeedJournal, _parse_utc, _utc
from stockbot.swing.paper_ledger import SEKPaperLedger, PaperTrade

GENESIS = "0" * 64
MODEL = "SYNTHETIC_RETROSPECTIVE_NEXT_XSTO_OPEN"
SCHEMA = 1
Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class SyntheticIntent:
    source_session: date
    target_session: date
    decided_at_utc: datetime
    source_entry_sha256: str
    symbol: str
    side: Side
    quantity: int
    commission_pct: float = 0.005
    min_commission_sek: float = 3.0
    source_kind: str = "SYNTHETIC_TEST_ONLY"
    # All-or-none optional synthetic BUY safety envelope. Old/manual
    # intents omit these fields for backwards-compatible journal replay.
    max_buy_notional_sek: float | None = None
    max_buy_risk_sek: float | None = None
    reference_stop_sek: float | None = None

    def __post_init__(self):
        if not isinstance(self.source_session, date) or isinstance(self.source_session, datetime):
            raise ValueError("invalid source session")
        if not isinstance(self.target_session, date) or isinstance(self.target_session, datetime):
            raise ValueError("invalid target session")
        if self.source_kind != "SYNTHETIC_TEST_ONLY":
            raise ValueError("EXTERNAL_SIGNALS_NOT_AUTHORIZED")
        if not re.fullmatch(r"[a-f0-9]{64}", self.source_entry_sha256):
            raise ValueError("invalid source feed SHA-256")
        if not isinstance(self.symbol, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", self.symbol):
            raise ValueError("invalid intent symbol")
        if self.side not in ("BUY", "SELL"):
            raise ValueError("invalid paper side")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) or not 1 <= self.quantity <= 10000:
            raise ValueError("invalid paper quantity")
        for v in (self.commission_pct, self.min_commission_sek):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v < 0:
                raise ValueError("invalid synthetic commission")
        if self.commission_pct > 0.2 or self.min_commission_sek > 1000:
            raise ValueError("unreasonable commission assumptions")
        envelope = (self.max_buy_notional_sek,
                    self.max_buy_risk_sek, self.reference_stop_sek)
        if any(v is not None for v in envelope):
            if self.side != "BUY" or any(
                isinstance(v, bool) or not isinstance(v, (int, float))
                or not isfinite(v) or v <= 0 for v in envelope
            ):
                raise ValueError("synthetic buy risk envelope must be complete and positive")
        if _utc(self.decided_at_utc) != self.decided_at_utc:
            raise ValueError("decision timestamp must be explicitly UTC")
        calendar = xcals.get_calendar("XSTO")
        source = pd.Timestamp(self.source_session)
        if not calendar.is_session(source):
            raise ValueError("source is not XSTO session")
        target = calendar.next_session(source)
        if self.target_session != target.date():
            raise ValueError("target must be next XSTO session")
        if _utc(self.decided_at_utc) >= calendar.session_open(target).to_pydatetime():
            raise ValueError("decision created after target market opened")

    def to_dict(self):
        data = asdict(self)
        data.update(
            source_session=self.source_session.isoformat(),
            target_session=self.target_session.isoformat(),
            decided_at_utc=self.decided_at_utc.isoformat(),
        )
        for key in ("max_buy_notional_sek", "max_buy_risk_sek", "reference_stop_sek"):
            if data[key] is None:
                del data[key]  # byte-compatible with existing v1 journal records
        return data

    @classmethod
    def from_dict(cls, raw):
        keys = set(cls.__dataclass_fields__)
        required = keys - {
            "commission_pct", "min_commission_sek", "source_kind",
            "max_buy_notional_sek", "max_buy_risk_sek", "reference_stop_sek",
        }
        if not isinstance(raw, dict) or not required.issubset(raw) or set(raw) - keys:
            raise ValueError("invalid synthetic intent schema")
        try:
            data = raw.copy()
            data["source_session"] = date.fromisoformat(data["source_session"])
            data["target_session"] = date.fromisoformat(data["target_session"])
            data["decided_at_utc"] = _parse_utc(data["decided_at_utc"])
            return cls(**data)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("invalid synthetic intent values") from exc

    @property
    def decision_id(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SyntheticExecution:
    mode: str
    decision_id: str
    source_session: str
    target_session: str
    status: str
    recorded_fill_price_sek: float | None
    fee_sek: float | None
    paper_cash_sek: float
    paper_positions: dict[str, int]
    real_forward_paper_started: bool = False
    broker_orders_sent: int = 0


class SyntheticPaperWorkflow:
    """Durable one-intent-per-source-session controlled synthetic workflow."""

    def __init__(self, *, feed_path: str | Path, intents_path: str | Path,
                 fills_path: str | Path, initial_sek: float = 1000.0):
        self.feed = PaperFeedJournal(feed_path)
        self.intents_path = Path(intents_path)
        self.lock_path = self.intents_path.with_name(self.intents_path.name + ".lock")
        self.ledger = SEKPaperLedger(fills_path, initial_cash_sek=initial_sek)
        if len({self.feed.path.resolve(), self.intents_path.resolve(),
                self.ledger.path.resolve()}) != 3:
            raise ValueError("feed, decisions and fills must use separate paths")

    @staticmethod
    def _hash(seq, previous, entry):
        canonical = json.dumps(
            {"version": SCHEMA, "sequence": seq, "previous_hash": previous,
             "intent": entry}, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return sha256(canonical).hexdigest()

    def _replay_intents(self):
        if not self.intents_path.exists():
            return []
        records = []
        prior = GENESIS
        previous_source = None
        try:
            lines = self.intents_path.read_text(encoding="utf-8").splitlines()
            for seq, line in enumerate(lines, 1):
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != {
                    "version", "sequence", "previous_hash", "entry_hash", "intent"
                }:
                    raise ValueError("invalid decision journal record")
                intent = SyntheticIntent.from_dict(row["intent"])
                if row["intent"] != intent.to_dict():
                    raise ValueError("noncanonical synthetic intent")
                if row["version"] != SCHEMA or row["sequence"] != seq or row["previous_hash"] != prior:
                    raise ValueError("decision chain order failure")
                if previous_source is not None and intent.source_session <= previous_source:
                    raise ValueError("duplicate/out-of-order signal sessions")
                expect = self._hash(seq, prior, row["intent"])
                if expect != row["entry_hash"]:
                    raise ValueError("decision journal hash mismatch")
                records.append(row)
                prior = expect
                previous_source = intent.source_session
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, KeyError) as exc:
            raise ValueError("invalid decision journal; fail closed") from exc
        return records

    def _verified_feed(self):
        # File locking and SHA replay performed by the existing feed journal.
        records = self.feed.verified_records()
        history = {date.fromisoformat(row["session"]["session"]): row for row in records}
        if records:
            days = pd.DatetimeIndex(sorted(history))
            cal = xcals.get_calendar("XSTO")
            expected = pd.DatetimeIndex(cal.sessions_in_range(
                days[0], days[-1]
            )).tz_localize(None).normalize()
            if not days.equals(expected):
                raise ValueError("INCOMPLETE_XSTO_FEED_HISTORY")
        return history

    @staticmethod
    def _find_bar(entry, symbol):
        for bar in entry["session"]["bars"]:
            if bar["symbol"] == symbol:
                return bar
        return None

    @staticmethod
    def _risk_rejection(intent: SyntheticIntent, opening_price: float) -> str | None:
        """Check the original EOD risk envelope against the NEXT market open.

        This is research protection, not a real guaranteed stop order.
        It cannot prevent losses when the next session gaps past a stop.
        """
        if intent.max_buy_notional_sek is None:
            return None  # legacy/manual research-only intent
        # This synthetic entry is an OPEN-only conditional limit/rejection
        # derived at EOD. If the market opens below the stop reference,
        # there is no valid new long entry at that open.
        if opening_price <= intent.reference_stop_sek:
            return "REJECTED_NEXT_OPEN_BELOW_PLANNED_STOP"
        notional = opening_price * intent.quantity
        if notional > intent.max_buy_notional_sek + 1e-8:
            return "REJECTED_NEXT_OPEN_POSITION_LIMIT"
        fee_at_buy = max(intent.min_commission_sek,
                         opening_price * intent.quantity * intent.commission_pct)
        fee_at_reference_exit = max(intent.min_commission_sek,
            intent.reference_stop_sek * intent.quantity * intent.commission_pct)
        planned_risk_with_fees = (
            intent.quantity*(opening_price-intent.reference_stop_sek)
            + fee_at_buy + fee_at_reference_exit
        )
        if planned_risk_with_fees > intent.max_buy_risk_sek + 1e-8:
            return "REJECTED_NEXT_OPEN_STOP_RISK"
        return None

    def plan(self, *, symbol: str, side: Side, quantity: int, clock_utc: datetime,
             commission_pct: float = 0.005, min_commission_sek: float = 3.0,
             expected_source_entry_sha256: str | None = None,
             expected_source_session: date | None = None,
             max_buy_notional_sek: float | None = None,
             max_buy_risk_sek: float | None = None,
             reference_stop_sek: float | None = None):
        """Persist before next open. Never observes the next session's bar."""
        clock = _utc(clock_utc)
        self.intents_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                intents = self._replay_intents()
                feed = self._verified_feed()
                if not feed:
                    raise ValueError("no prior synthetic daily session")
                source_day = max(feed)
                source = feed[source_day]
                # Auto-signal preview is only useful if the exact source
                # record remains unchanged during the planning transaction.
                if (expected_source_entry_sha256 is not None
                    and expected_source_entry_sha256 != source["entry_hash"]):
                    raise ValueError("SIGNAL_SOURCE_HASH_CHANGED")
                if (expected_source_session is not None
                    and expected_source_session != source_day):
                    raise ValueError("SIGNAL_SOURCE_SESSION_CHANGED")
                if _parse_utc(source["session"]["observed_at_utc"]) > clock:
                    raise ValueError("source bar was not observed at decision time")
                if self._find_bar(source, symbol) is None:
                    raise ValueError("symbol absent from latest observed daily bar")
                # Repeated client calls after a crash return the original
                # durable decision, even with a newer wall-clock timestamp.
                # Reject any change of trade economics for the same session.
                if intents and date.fromisoformat(intents[-1]["intent"]["source_session"]) == source_day:
                    existing = SyntheticIntent.from_dict(intents[-1]["intent"])
                    if (existing.source_entry_sha256 == source["entry_hash"]
                        and existing.symbol == symbol
                        and existing.side == side
                        and existing.quantity == quantity
                        and existing.commission_pct == commission_pct
                        and existing.min_commission_sek == min_commission_sek
                        and existing.max_buy_notional_sek == max_buy_notional_sek
                        and existing.max_buy_risk_sek == max_buy_risk_sek
                        and existing.reference_stop_sek == reference_stop_sek):
                        return existing.decision_id
                    raise ValueError("CONFLICTING_DECISION_FOR_SESSION")
                target = xcals.get_calendar("XSTO").next_session(
                    pd.Timestamp(source_day)
                ).date()
                intent = SyntheticIntent(
                    source_session=source_day, target_session=target,
                    decided_at_utc=clock, source_entry_sha256=source["entry_hash"],
                    symbol=symbol, side=side, quantity=quantity,
                    commission_pct=commission_pct, min_commission_sek=min_commission_sek,
                    max_buy_notional_sek=max_buy_notional_sek,
                    max_buy_risk_sek=max_buy_risk_sek,
                    reference_stop_sek=reference_stop_sek,
                )
                if intents and date.fromisoformat(intents[-1]["intent"]["source_session"]) >= source_day:
                    raise ValueError("OUT_OF_ORDER_DECISION_SESSION")
                # A synthetic sell may only be planned against shares already
                # held in the replay-verified SEK ledger. Never create an
                # unfulfillable naked-sale intent and block future sessions.
                if side == "SELL" and quantity > self.ledger.snapshot().positions.get(symbol, 0):
                    raise ValueError("NAKED_PAPER_SELL_AT_DECISION")
                # Don't create new intents when earlier decided synthetic fills
                # are still unaccounted despite the target session being present.
                for row in intents:
                    pending = SyntheticIntent.from_dict(row["intent"])
                    if pending.target_session in feed and not self.ledger.has_decision(pending.decision_id):
                        target_bar = self._find_bar(feed[pending.target_session], pending.symbol)
                        # A rejected EOD-specified buy is terminal; never
                        # write a fill and never block later paper sessions.
                        if (target_bar is not None and target_bar["volume"] > 0
                            and self._risk_rejection(pending, float(target_bar["open"]))):
                            continue
                        raise ValueError("SETTLE_PREVIOUS_DECISION_FIRST")
                seq = len(intents) + 1
                prior = intents[-1]["entry_hash"] if intents else GENESIS
                data = intent.to_dict()
                new_record = {
                    "version": SCHEMA, "sequence": seq, "previous_hash": prior,
                    "entry_hash": self._hash(seq, prior, data), "intent": data,
                }
                all_rows = [*intents, new_record]
                serialized = "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                    for row in all_rows
                )
                tmp = self.intents_path.with_name(
                    f".{self.intents_path.name}.{uuid4().hex}.tmp"
                )
                try:
                    with tmp.open("x", encoding="utf-8") as fh:
                        fh.write(serialized)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp, self.intents_path)
                    directory_fd = os.open(self.intents_path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                finally:
                    tmp.unlink(missing_ok=True)
                return intent.decision_id
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def reconcile(self, *, clock_utc: datetime) -> tuple[SyntheticExecution, ...]:
        """Reconstruct a next-open synthetic price only after next EOD bar.

        A crash after a ledger write needs no second mutation: ledger IDs
        recover the settled state. Never interpret this as an auction fill.
        """
        clock = _utc(clock_utc)
        self.intents_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                intents = self._replay_intents()
                feed = self._verified_feed()
                results = []
                for row in intents:
                    intent = SyntheticIntent.from_dict(row["intent"])
                    source = feed.get(intent.source_session)
                    if source is None or source["entry_hash"] != intent.source_entry_sha256:
                        raise ValueError("DECISION_SOURCE_FEED_REWRITTEN")
                    target = feed.get(intent.target_session)
                    if target is None and any(day > intent.target_session for day in feed):
                        raise ValueError("MISSING_NEXT_SESSION_DATA")
                    if target is None or _parse_utc(target["session"]["observed_at_utc"]) > clock:
                        status = "AWAITING_NEXT_COMPLETED_SESSION"
                        price, fee = None, None
                    else:
                        bar = self._find_bar(target, intent.symbol)
                        if bar is None or bar["volume"] <= 0:
                            raise ValueError("NEXT_SESSION_MISSING_OR_HALTED_SYMBOL")
                        price = float(bar["open"])
                        fee = round(max(
                            intent.min_commission_sek,
                            price * intent.quantity * intent.commission_pct
                        ), 6)
                        trade = PaperTrade(
                            decision_id=intent.decision_id,
                            timestamp=_parse_utc(target["session"]["observed_at_utc"]),
                            symbol=intent.symbol, side=intent.side,
                            quantity=intent.quantity, price_sek=price, fee_sek=fee
                        )
                        previously = self.ledger.find_decision(intent.decision_id)
                        rejection = self._risk_rejection(intent, price)
                        if rejection:
                            if previously is not None:
                                raise ValueError("RISK_REJECTED_BUT_LEDGER_HAS_FILL")
                            status, price, fee = rejection, None, None
                        elif previously is not None:
                            if previously != trade:
                                raise ValueError("PAPER_FILL_CONFLICT_ON_RECOVERY")
                            status = "RECONSTRUCTED_PREVIOUSLY"
                        else:
                            self.ledger.record_paper_fill(trade)
                            status = "RETROSPECTIVE_SYNTHETIC_FILL"
                    balance = self.ledger.snapshot()
                    results.append(SyntheticExecution(
                        mode=MODEL, decision_id=intent.decision_id,
                        source_session=intent.source_session.isoformat(),
                        target_session=intent.target_session.isoformat(),
                        status=status, recorded_fill_price_sek=price, fee_sek=fee,
                        paper_cash_sek=round(balance.cash_sek, 6),
                        paper_positions=balance.positions,
                    ))
                return tuple(results)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
