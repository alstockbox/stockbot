from datetime import datetime, timezone
import json

import pytest

from stockbot.journal.trade_journal import AppendOnlyTradeJournal, DecisionEvent
from stockbot.opportunity.ranker import (
    Opportunity,
    OpportunityPolicy,
    rank_opportunities,
)


def event(decision_id: str) -> DecisionEvent:
    return DecisionEvent(
        decision_id=decision_id,
        timestamp=datetime(2026, 9, 26, tzinfo=timezone.utc),
        symbol="EURUSD",
        strategy="trend",
        regime="bull_trend",
        signal_score=0.8,
        confidence=0.7,
        expected_return=0.02,
        expected_cost=0.001,
        expected_risk=0.01,
        approved=True,
    )


def test_trade_journal_is_append_only_and_hash_chained(tmp_path):
    path = tmp_path / "journal.jsonl"
    journal = AppendOnlyTradeJournal(path)
    first = journal.append(event("d1"))
    second = journal.append(event("d2"))

    rows = journal.entries()
    assert [row.sequence for row in rows] == [1, 2]
    assert second.previous_hash == first.entry_hash


def test_trade_journal_detects_tampering(tmp_path):
    path = tmp_path / "journal.jsonl"
    journal = AppendOnlyTradeJournal(path)
    journal.append(event("d1"))
    row = json.loads(path.read_text())
    row["event"]["expected_return"] = 99.0
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="integrity"):
        journal.entries()


def test_opportunity_ranker_prefers_net_edge_not_raw_return():
    high_raw_bad_risk = Opportunity(
        "A", "model", 0.08, 0.01, 0.05, 0.02, 0.02, 0.9
    )
    lower_raw_clean = Opportunity(
        "B", "model", 0.04, 0.002, 0.005, 0.002, 0.003, 0.9
    )
    ranked = rank_opportunities([high_raw_bad_risk, lower_raw_clean])
    assert ranked[0].opportunity.symbol == "B"


def test_ranker_can_take_zero_trades_when_no_edge_exists():
    weak = Opportunity(
        "A", "model", 0.005, 0.002, 0.004, 0.001, 0.001, 0.7
    )
    ranked = rank_opportunities(
        [weak],
        OpportunityPolicy(minimum_net_edge=0.001),
    )
    assert ranked == []
