"""Synthetic lifecycle replay tests: IPO, bankruptcy and cash settlements."""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import exchange_calendars as xcals
import pandas as pd
import pytest

from stockbot.swing.engine import SwingConfig, run_swing_backtest
from stockbot.swing.lifecycle import DelistingSettlement, parse_settlements
from stockbot.swing.universe import InstrumentHistory

ASOF = "2026-10-29T22:00:00Z"


def specimen():
    days = xcals.get_calendar("XSTO").sessions_in_range(
        "2025-01-02", "2026-05-01"
    ).tz_localize(None)[:265]
    rows = []
    for symbol, eligible in (("AAA", days), ("BBB", days[20:246])):
        for i, day in enumerate(eligible):
            p = 30.0 if symbol == "AAA" else 20.0 + i*.11
            rows.append({
                "date": day, "symbol": symbol,
                "open":p-.04, "high":p+.05, "low":p-.12,
                "close":p, "volume":20000,
            })
    frame = pd.DataFrame(rows)
    history = (
        InstrumentHistory("AAA", "SE0000000001", days[0].date(), None,
                          "2024-12-01T00:00:00Z", None, "ACTIVE"),
        InstrumentHistory("BBB", "SE0000000002", days[20].date(),
                          days[245].date(), "2025-01-01T00:00:00Z",
                          "2025-01-01T00:00:00Z", "DELISTED"),
    )
    settlement = DelistingSettlement(
        "BBB", days[245].date(), 2.0, days[259].date(),
        (days[245]-pd.Timedelta(days=7)).strftime("%Y-%m-%dT07:00:00Z"),
        "SYNTHETIC_TEST_CASH_RECEIPT",
    )
    return frame, days, history, settlement


def test_ipo_delist_cash_only_removes_position_without_phantom_market_sale():
    frame, days, records, settlement = specimen()
    r=run_swing_backtest(frame, universe=records, settlements=(settlement,),
                         collected_at_utc=ASOF)
    bbb_buys=[f for f in r.fills if f.symbol=="BBB" and f.side=="BUY"]
    assert bbb_buys, "need an actual held investment to test delisting"
    assert all(f.date >= str(days[220].date()) for f in bbb_buys)
    assert r.delisted_shares > 0
    assert r.lifecycle_events == (f"{days[245].date()}:BBB:DELISTED_CASH_ONLY",)
    assert r.delist_cash_paid_sek > 0
    assert r.delist_receivable_sek == 0
    assert not any(f.reason == "DELISTED_CASH_ONLY" for f in r.fills)
    assert all(f.date <= str(days[245].date()) for f in r.fills if f.symbol=="BBB")


def test_zero_recovery_writes_off_held_shares_not_last_quote_sale():
    frame, days, records, settlement=specimen()
    r=run_swing_backtest(frame, universe=records,
                         settlements=(replace(settlement,cash_per_share_sek=0.0),),
                         collected_at_utc=ASOF)
    assert r.delisted_shares>0
    assert r.delist_cash_paid_sek==0
    assert r.delist_receivable_sek==0
    assert r.final_equity_sek < 1_000.0
    assert not any(f.symbol=="BBB" and f.side=="SELL" and
                   f.date==str(days[245].date()) for f in r.fills)


def test_missing_or_misdated_settlement_fails_closed():
    frame, days, records, settlement=specimen()
    with pytest.raises(ValueError,match="missing or unexpected"):
        run_swing_backtest(frame,universe=records,collected_at_utc=ASOF)
    with pytest.raises(ValueError,match="last session mismatches"):
        run_swing_backtest(frame,universe=records,
                           settlements=(replace(settlement,last_session=days[244].date()),),
                           collected_at_utc=ASOF)


def test_outstanding_cash_settlement_stays_in_nav_not_tradable_cash():
    frame, days, records, settlement=specimen()
    delayed=replace(settlement,pay_date=days[-1].date()+timedelta(days=90))
    r=run_swing_backtest(frame, universe=records, settlements=(delayed,),
                         collected_at_utc=ASOF)
    assert r.delisted_shares>0
    assert r.delist_receivable_sek == pytest.approx(r.delisted_shares*2.0)
    assert r.delist_cash_paid_sek == 0


def test_an_ipo_never_trades_before_listing_or_first_200_available_bars():
    frame, days, records, settlement=specimen()
    r=run_swing_backtest(frame,universe=records,settlements=(settlement,),
                         collected_at_utc=ASOF)
    assert all(f.date >= str(days[220].date()) for f in r.fills if f.symbol=="BBB")


def test_bad_settlement_schema_and_noncash_blocked():
    frame, days, records, settlement=specimen()
    with pytest.raises(ValueError,match="only explicit SEK cash"):
        replace(settlement, settlement_kind="SHARE_EXCHANGE")
    with pytest.raises(ValueError,match="nonnegative finite"):
        replace(settlement,cash_per_share_sek=float("nan"))
    with pytest.raises(ValueError,match="after final"):
        replace(settlement,pay_date=settlement.last_session)
    with pytest.raises(ValueError,match="JSON list"):
        parse_settlements({"settlements":[]})
    with pytest.raises(ValueError,match="invalid delisting settlement schema"):
        parse_settlements([{"symbol":"BBB","extra":"unknown"}])


def test_no_replay_without_asof_even_with_history():
    frame, days, records, settlement=specimen()
    with pytest.raises(ValueError,match="collection as-of"):
        run_swing_backtest(frame,universe=records,settlements=(settlement,))
    with pytest.raises(ValueError,match="require audited"):
        run_swing_backtest(frame,settlements=(settlement,))

def test_delisting_info_cannot_depend_on_last_session_future_close():
    frame, days, records, settlement=specimen()
    with pytest.raises(ValueError,match="before final session open"):
        replace(settlement,known_at_utc=days[245].strftime("%Y-%m-%dT17:00:00Z"))
