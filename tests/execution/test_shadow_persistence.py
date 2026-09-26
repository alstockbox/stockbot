from datetime import datetime, timedelta, timezone

from stockbot.data.live_models import AccountSnapshot, MarketQuote, SymbolSpec
from stockbot.opportunity.ranker import Opportunity
from stockbot.runtime.shadow_runtime import ShadowTradePlan, ShadowTradingRuntime


class MutableProvider:
    def __init__(self, now: datetime):
        self.now = now
        self.bid = 1.1000
        self.ask = 1.1002
        self.quote_timestamp = now

    def account_snapshot(self, *, now=None):
        stamp = self.now if now is None else now
        return AccountSnapshot(466.0, 466.0, 466.0, stamp, "USD")

    def quote(self, symbol):
        return MarketQuote(
            symbol,
            self.bid,
            self.ask,
            self.quote_timestamp,
            "mutable",
        )

    def symbol_spec(self, symbol):
        return SymbolSpec(
            symbol,
            0.00001,
            1.0,
            0.01,
            100.0,
            0.01,
            True,
        )


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


def plan(decision_id="a", symbol="EURUSD"):
    return ShadowTradePlan(
        decision_id,
        opportunity(symbol=symbol, source=f"model-{decision_id}"),
        0.003,
        0.25,
    )


def test_shadow_position_survives_runtime_restart(tmp_path):
    now = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)
    provider = MutableProvider(now)
    journal = tmp_path / "journal.jsonl"

    first = ShadowTradingRuntime(provider, journal)
    result = first.run_cycle(
        [plan()],
        timestamp=now,
        regime="neutral_chop",
    )
    assert result.accepted == 1
    assert len(first.executor.positions) == 1

    second = ShadowTradingRuntime(provider, journal)
    assert len(second.executor.positions) == 1
    assert second.executor.positions[0].decision_id == "a"
    assert second.portfolio_state is not None
    assert len(second.portfolio_state.positions) == 1


def test_stop_loss_exit_is_persisted_and_journaled(tmp_path):
    now = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)
    provider = MutableProvider(now)
    journal = tmp_path / "journal.jsonl"

    runtime = ShadowTradingRuntime(provider, journal)
    opened = runtime.run_cycle(
        [plan()],
        timestamp=now,
        regime="neutral_chop",
    )
    assert opened.accepted == 1
    stop = runtime.executor.positions[0].stop_loss

    provider.bid = stop * 0.999
    provider.ask = provider.bid * 1.0001
    provider.quote_timestamp = now + timedelta(seconds=1)

    restarted = ShadowTradingRuntime(provider, journal)
    refresh = restarted.refresh_positions(
        timestamp=now + timedelta(seconds=1),
    )

    assert refresh.complete
    assert len(refresh.exits) == 1
    assert refresh.exits[0].reason == "STOP_LOSS"
    assert refresh.exits[0].realized_pnl_usd < 0.0
    assert len(restarted.executor.positions) == 0
    assert restarted.portfolio_state is not None
    assert restarted.portfolio_state.realized_pnl_usd < 0.0
    assert restarted.daily_realized_pnl(now) < 0.0

    third = ShadowTradingRuntime(provider, journal)
    assert len(third.executor.positions) == 0
    assert third.portfolio_state is not None
    assert third.portfolio_state.realized_pnl_usd < 0.0


def test_stale_open_position_mark_blocks_new_entries(tmp_path):
    now = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)
    provider = MutableProvider(now)
    journal = tmp_path / "journal.jsonl"
    runtime = ShadowTradingRuntime(provider, journal)

    first = runtime.run_cycle(
        [plan("a")],
        timestamp=now,
        regime="neutral_chop",
    )
    assert first.accepted == 1

    later = now + timedelta(seconds=40)
    provider.quote_timestamp = now
    second = runtime.run_cycle(
        [plan("b")],
        timestamp=later,
        regime="neutral_chop",
    )

    assert second.accepted == 0
    assert second.rejected == 1
    assert second.decisions[0].reasons == (
        "PORTFOLIO_MARK_UNAVAILABLE",
    )
    assert len(runtime.executor.positions) == 1
