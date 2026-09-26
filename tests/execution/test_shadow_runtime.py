from datetime import datetime, timedelta, timezone

from stockbot.data.live_models import AccountSnapshot, MarketQuote, SymbolSpec
from stockbot.opportunity.ranker import Opportunity, OpportunityPolicy
from stockbot.runtime.shadow_runtime import ShadowTradePlan, ShadowTradingRuntime


class FakeProvider:
    def __init__(self, now: datetime, *, bid=1.1000, ask=1.1002, quote_age=0.0):
        self.now = now
        self.bid = bid
        self.ask = ask
        self.quote_age = quote_age

    def account_snapshot(self, *, now=None):
        stamp = self.now if now is None else now
        return AccountSnapshot(466.0, 466.0, 466.0, stamp, "USD")

    def quote(self, symbol):
        return MarketQuote(
            symbol,
            self.bid,
            self.ask,
            self.now - timedelta(seconds=self.quote_age),
            "fake",
        )

    def symbol_spec(self, symbol):
        return SymbolSpec(symbol, 0.00001, 1.0, 0.01, 100.0, 0.01, True)


def opportunity(symbol="EURUSD", source="model"):
    return Opportunity(
        symbol=symbol,
        source=source,
        expected_return=0.01,
        expected_cost=0.0001,
        downside_risk=0.001,
        correlation_penalty=0.001,
        uncertainty=0.001,
        confidence=0.9,
    )


def test_shadow_runtime_uses_live_quote_and_opens_no_broker_order(tmp_path):
    now = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)
    runtime = ShadowTradingRuntime(
        FakeProvider(now),
        tmp_path / "journal.jsonl",
        opportunity_policy=OpportunityPolicy(minimum_net_edge=0.0005),
    )
    result = runtime.run_cycle(
        [ShadowTradePlan("a", opportunity(), 0.003, 0.25)],
        timestamp=now,
        regime="neutral_chop",
    )

    assert result.accepted == 1
    assert result.rejected == 0
    assert len(runtime.executor.positions) == 1
    position = runtime.executor.positions[0]
    assert position.entry_price == 1.1002
    assert len(runtime.journal.entries()) == 1


def test_stale_quote_is_rejected_before_shadow_execution(tmp_path):
    now = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)
    runtime = ShadowTradingRuntime(
        FakeProvider(now, quote_age=31.0),
        tmp_path / "journal.jsonl",
    )
    result = runtime.run_cycle(
        [ShadowTradePlan("a", opportunity(), 0.003, 0.25)],
        timestamp=now,
        regime="neutral_chop",
    )

    assert result.accepted == 0
    assert result.rejected == 1
    assert "STALE_QUOTE" in result.decisions[0].reasons
    assert len(runtime.executor.positions) == 0


def test_actual_spread_can_remove_expected_edge(tmp_path):
    now = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)
    runtime = ShadowTradingRuntime(
        FakeProvider(now, bid=1.0, ask=1.015),
        tmp_path / "journal.jsonl",
        opportunity_policy=OpportunityPolicy(minimum_net_edge=0.0005),
    )
    result = runtime.run_cycle(
        [ShadowTradePlan("a", opportunity(), 0.003, 0.25)],
        timestamp=now,
        regime="neutral_chop",
    )

    assert result.accepted == 0
    assert "BELOW_MINIMUM_NET_EDGE" in result.decisions[0].reasons


def test_total_open_risk_accumulates_across_shadow_positions(tmp_path):
    now = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)
    runtime = ShadowTradingRuntime(
        FakeProvider(now),
        tmp_path / "journal.jsonl",
    )
    plans = [
        ShadowTradePlan(
            f"d{index}",
            opportunity(symbol=f"S{index}", source=f"m{index}"),
            0.005,
            0.20,
        )
        for index in range(5)
    ]
    result = runtime.run_cycle(
        plans,
        timestamp=now,
        regime="bull_trend",
    )

    assert result.accepted == 4
    assert result.rejected == 1
    rejected = [decision for decision in result.decisions if not decision.approved]
    assert rejected[0].reasons == ("TOTAL_OPEN_RISK_LIMIT",)


def test_duplicate_decision_id_is_rejected_without_duplicate_journal_entry(tmp_path):
    now = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)
    runtime = ShadowTradingRuntime(
        FakeProvider(now),
        tmp_path / "journal.jsonl",
    )
    plan = ShadowTradePlan("same", opportunity(), 0.003, 0.25)

    first = runtime.run_cycle([plan], timestamp=now, regime="neutral_chop")
    second = runtime.run_cycle([plan], timestamp=now, regime="neutral_chop")

    assert first.accepted == 1
    assert second.accepted == 0
    assert second.decisions[0].reasons == ("DUPLICATE_DECISION_ID",)
    assert len(runtime.journal.entries()) == 1
