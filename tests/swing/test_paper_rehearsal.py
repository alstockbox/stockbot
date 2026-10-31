"""Broker-isolated, synthetic EOD decision and retrospective OPEN fill tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import exchange_calendars as xcals
import pandas as pd
import pytest

from stockbot.swing.paper_feed import PaperBar, PaperSession, PaperFeedJournal
from stockbot.swing.paper_ledger import SEKPaperLedger
from stockbot.swing.paper_rehearsal import SyntheticPaperWorkflow, MODEL


def dates():
    c=xcals.get_calendar("XSTO")
    d1=pd.Timestamp("2025-01-02")
    d2=c.next_session(d1)
    d3=c.next_session(d2)
    return [d1.date(),d2.date(),d3.date()]


def observed(day):
    c=xcals.get_calendar("XSTO")
    return c.session_close(pd.Timestamp(day)).to_pydatetime()+timedelta(minutes=20)


def session(day, price=100, *, volume=10000):
    return PaperSession(
        session=day, observed_at_utc=observed(day),
        source_kind="SYNTHETIC_TEST_ONLY",
        source_reference="synthetic:decision_rehearsal",
        bars=(PaperBar("AAA",price,price+2,price-2,price+1,volume),)
    )


def env(tmp_path):
    feed=tmp_path/"feed.jsonl"
    intents=tmp_path/"intents.jsonl"
    fills=tmp_path/"fills.jsonl"
    work=SyntheticPaperWorkflow(
        feed_path=feed,intents_path=intents,fills_path=fills
    )
    return work,PaperFeedJournal(feed),SEKPaperLedger(fills)


def stage(feed, day, price=100, volume=10000):
    event=session(day,price,volume=volume)
    feed.stage(event,clock_utc=event.observed_at_utc+timedelta(minutes=1))
    return event


def test_decide_first_reconstruct_after_next_close_then_restart(tmp_path):
    w,f,ledger=env(tmp_path)
    first,second,third=dates()
    a=stage(f,first,100)
    now=a.observed_at_utc+timedelta(minutes=3)
    intent_id=w.plan(symbol="AAA",side="BUY",quantity=2,clock_utc=now)
    assert len(intent_id)==64
    assert not ledger.has_decision(intent_id)
    wait=w.reconcile(clock_utc=now)
    assert wait[0].status=="AWAITING_NEXT_COMPLETED_SESSION"
    assert wait[0].recorded_fill_price_sek is None
    assert wait[0].paper_cash_sek==1000
    assert wait[0].broker_orders_sent==0
    b=stage(f,second,105)
    after=b.observed_at_utc+timedelta(minutes=4)
    fills=w.reconcile(clock_utc=after)
    assert fills[0].status=="RETROSPECTIVE_SYNTHETIC_FILL"
    assert fills[0].mode==MODEL
    assert fills[0].recorded_fill_price_sek==105
    assert fills[0].fee_sek==3
    assert fills[0].paper_cash_sek==787
    assert fills[0].paper_positions=={"AAA":2}
    assert not fills[0].real_forward_paper_started
    assert ledger.snapshot().last_sequence==1
    restarted=SyntheticPaperWorkflow(feed_path=f.path,intents_path=w.intents_path,
                                      fills_path=ledger.path)
    recovered=restarted.reconcile(clock_utc=after)
    assert recovered[0].status=="RECONSTRUCTED_PREVIOUSLY"
    assert recovered[0].paper_cash_sek==787
    assert ledger.snapshot().last_sequence==1


def test_next_day_sell_and_exact_stockholm_calendar(tmp_path):
    w,f,ledger=env(tmp_path)
    first,second,third=dates()
    assert str(third)=="2025-01-07"  # Jan 6 Swedish bank holiday
    a=stage(f,first)
    w.plan(symbol="AAA",side="BUY",quantity=2,
           clock_utc=a.observed_at_utc+timedelta(minutes=2))
    b=stage(f,second,105)
    w.reconcile(clock_utc=b.observed_at_utc+timedelta(minutes=3))
    second_id=w.plan(symbol="AAA",side="SELL",quantity=1,
                     clock_utc=b.observed_at_utc+timedelta(minutes=4))
    assert ledger.find_decision(second_id) is None
    c=stage(f,third,106)
    r=w.reconcile(clock_utc=c.observed_at_utc+timedelta(minutes=5))
    assert [x.status for x in r]==["RECONSTRUCTED_PREVIOUSLY","RETROSPECTIVE_SYNTHETIC_FILL"]
    assert r[-1].recorded_fill_price_sek==106
    assert r[-1].paper_cash_sek==890
    assert r[-1].paper_positions=={"AAA":1}
    assert ledger.snapshot().last_sequence==2


def test_replay_and_same_decision_idempotent_conflict_is_blocked(tmp_path):
    w,f,ledger=env(tmp_path)
    a=stage(f,dates()[0])
    clock=a.observed_at_utc+timedelta(minutes=1)
    d=w.plan(symbol="AAA",side="BUY",quantity=1,clock_utc=clock)
    assert w.plan(symbol="AAA",side="BUY",quantity=1,clock_utc=clock+timedelta(minutes=1))==d
    assert len(w.intents_path.read_text().splitlines())==1
    with pytest.raises(ValueError,match="CONFLICTING_DECISION"):
        w.plan(symbol="AAA",side="SELL",quantity=1,clock_utc=clock)


def test_wrong_time_future_data_and_no_signal_without_source(tmp_path):
    w,f,ledger=env(tmp_path)
    with pytest.raises(ValueError,match="no prior"):
        w.plan(symbol="AAA",side="BUY",quantity=1,clock_utc=datetime(2026,1,1,tzinfo=timezone.utc))
    a=stage(f,dates()[0])
    with pytest.raises(ValueError,match="not observed"):
        w.plan(symbol="AAA",side="BUY",quantity=1,
               clock_utc=a.observed_at_utc-timedelta(seconds=1))
    target_open=xcals.get_calendar("XSTO").session_open(
        pd.Timestamp(dates()[1])
    ).to_pydatetime()
    with pytest.raises(ValueError,match="after target market opened"):
        w.plan(symbol="AAA",side="BUY",quantity=1,clock_utc=target_open)


def test_overspend_and_naked_sale_never_record_fills(tmp_path):
    w,f,ledger=env(tmp_path)
    a=stage(f,dates()[0])
    w.plan(symbol="AAA",side="BUY",quantity=50,clock_utc=a.observed_at_utc+timedelta(minutes=1))
    b=stage(f,dates()[1],500)
    with pytest.raises(ValueError,match="INSUFFICIENT_PAPER_CASH"):
        w.reconcile(clock_utc=b.observed_at_utc+timedelta(minutes=3))
    assert ledger.snapshot().last_sequence==0
    assert ledger.snapshot().cash_sek==1000


def test_corrupted_source_or_intent_refuses_reconciliation(tmp_path):
    w,f,ledger=env(tmp_path)
    a=stage(f,dates()[0])
    w.plan(symbol="AAA",side="BUY",quantity=1,clock_utc=a.observed_at_utc+timedelta(minutes=1))
    b=stage(f,dates()[1])
    p=w.intents_path
    saved=p.read_text()
    p.write_text(saved.replace('"quantity":1','"quantity":2'))
    with pytest.raises(ValueError,match="hash mismatch"):
        w.reconcile(clock_utc=b.observed_at_utc+timedelta(minutes=1))
    p.write_text(saved)
    p=f.path
    raw=p.read_text()
    assert raw.count("synthetic:decision_rehearsal")==2
    p.write_text(raw.replace("synthetic:decision_rehearsal","synthetic:altered_source"))
    with pytest.raises(ValueError,match="hash mismatch"):
        w.reconcile(clock_utc=b.observed_at_utc+timedelta(minutes=1))
    assert ledger.snapshot().last_sequence==0


def test_reconcile_cannot_reconstruct_future_target_session(tmp_path):
    w,f,ledger=env(tmp_path)
    a=stage(f,dates()[0])
    w.plan(symbol="AAA",side="BUY",quantity=1,clock_utc=a.observed_at_utc+timedelta(minutes=2))
    b=stage(f,dates()[1])
    assert w.reconcile(clock_utc=b.observed_at_utc-timedelta(minutes=1))[0].status=="AWAITING_NEXT_COMPLETED_SESSION"
    assert ledger.snapshot().last_sequence==0


def test_halted_next_session_stays_blocked(tmp_path):
    w,f,ledger=env(tmp_path)
    a=stage(f,dates()[0])
    w.plan(symbol="AAA",side="BUY",quantity=1,clock_utc=a.observed_at_utc+timedelta(minutes=1))
    b=stage(f,dates()[1],volume=0)
    with pytest.raises(ValueError,match="HALTED_SYMBOL"):
        w.reconcile(clock_utc=b.observed_at_utc+timedelta(minutes=1))
    assert ledger.snapshot().last_sequence==0


def test_ledger_read_api_verifies_entry_and_fails_closed(tmp_path):
    w,f,ledger=env(tmp_path)
    assert not ledger.has_decision("a")
    a=stage(f,dates()[0])
    decision=w.plan(symbol="AAA",side="BUY",quantity=1,clock_utc=a.observed_at_utc+timedelta(minutes=1))
    b=stage(f,dates()[1])
    w.reconcile(clock_utc=b.observed_at_utc+timedelta(minutes=1))
    assert ledger.has_decision(decision)
    p=ledger.path
    p.write_text(p.read_text().replace('"quantity": 1','"quantity": 2'))
    with pytest.raises(ValueError,match="hash mismatch"):
        ledger.find_decision(decision)

def test_repeated_decision_idempotent_even_after_target_market_open(tmp_path):
    w,f,ledger=env(tmp_path)
    a=stage(f,dates()[0])
    original=w.plan(symbol="AAA",side="BUY",quantity=1,
                    clock_utc=a.observed_at_utc+timedelta(minutes=1))
    target_open=xcals.get_calendar("XSTO").session_open(
        pd.Timestamp(dates()[1])
    ).to_pydatetime()
    assert w.plan(symbol="AAA",side="BUY",quantity=1,
                  clock_utc=target_open+timedelta(minutes=1))==original
    assert len(w.intents_path.read_text().splitlines())==1


def test_naked_sell_rejected_when_planning_not_deferred_to_fill(tmp_path):
    w,f,ledger=env(tmp_path)
    a=stage(f,dates()[0])
    with pytest.raises(ValueError,match="NAKED_PAPER_SELL_AT_DECISION"):
        w.plan(symbol="AAA",side="SELL",quantity=1,
               clock_utc=a.observed_at_utc+timedelta(minutes=1))
    assert not w.intents_path.exists()

def test_missing_xsto_trading_day_never_becomes_a_valid_rehearsal(tmp_path):
    w,f,ledger=env(tmp_path)
    first,second,third=dates()
    a=stage(f,first)
    w.plan(symbol="AAA",side="BUY",quantity=1,
           clock_utc=a.observed_at_utc+timedelta(minutes=1))
    stage(f,third)
    with pytest.raises(ValueError,match="INCOMPLETE_XSTO_FEED_HISTORY"):
        w.reconcile(clock_utc=observed(third)+timedelta(minutes=1))
    assert ledger.snapshot().last_sequence==0
