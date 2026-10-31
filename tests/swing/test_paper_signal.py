"""Causal synthetic-only EOD candidate tests: no future bars, no broker."""
from __future__ import annotations

from datetime import timedelta, date
from pathlib import Path
import shutil
import subprocess
import sys
import json

import exchange_calendars as xcals
import pandas as pd
import pytest

from stockbot.swing.engine import SwingConfig
from stockbot.swing.paper_feed import PaperBar, PaperSession, PaperFeedJournal
from stockbot.swing.paper_rehearsal import SyntheticPaperWorkflow
from stockbot.swing.paper_signal import SyntheticSwingSignal


@pytest.fixture(scope="module")
def source_fixture(tmp_path_factory):
    path=tmp_path_factory.mktemp("synthetic_signal_source")/"feed.jsonl"
    feed=PaperFeedJournal(path)
    calendar=xcals.get_calendar("XSTO")
    days=calendar.sessions_in_range("2025-01-02","2026-03-01").tz_localize(None)[:215]
    for i,day in enumerate(days):
        # Flat first 180 sessions, then rising, unambiguous SMA20/50/200
        # and BREAKOUT at latest close. No external market observations.
        close=50.0+max(0,i-179)*.32
        at=calendar.session_close(day).to_pydatetime()+timedelta(minutes=20)
        event=PaperSession(
            day.date(),at,"SYNTHETIC_TEST_ONLY",
            "synthetic:fully_offline_rolling_indicator_test",
            (PaperBar("AAA",close-.02,close+.12,close-.12,close,20000),)
        )
        feed.stage(event,clock_utc=at+timedelta(minutes=1))
    return path,days


def setup(tmp_path, source_fixture, cfg=None):
    src,days=source_fixture
    feed_path=tmp_path/"feed.jsonl"
    shutil.copyfile(src,feed_path)
    workflow=SyntheticPaperWorkflow(
        feed_path=feed_path,
        intents_path=tmp_path/"intents.jsonl",
        fills_path=tmp_path/"fills.jsonl"
    )
    engine=SyntheticSwingSignal(workflow,config=cfg)
    last_time=xcals.get_calendar("XSTO").session_close(
        days[-1]
    ).to_pydatetime()+timedelta(minutes=23)
    return engine,workflow,days,last_time


def test_signal_is_causal_and_risk_bounded(tmp_path,source_fixture):
    engine,work,days,now=setup(tmp_path,source_fixture)
    out=engine.preview(clock_utc=now)
    assert out.status=="SIGNAL"
    assert out.side=="BUY"
    assert out.symbol=="AAA"
    assert out.signal_reason=="BREAKOUT"
    assert out.quantity>=1
    assert out.technical_sessions==215
    assert out.estimated_risk_sek<=5.0+1e-6
    assert out.last_close_sek*out.quantity<=200.0+1e-6
    assert out.source_session==days[-1].date().isoformat()
    assert out.source_entry_sha256==work.feed.snapshot().chain_head_sha256
    assert out.staged_decision_id is None
    assert not work.intents_path.exists()
    assert not work.ledger.path.exists()
    assert not out.authorized_real_market_data
    assert not out.actual_forward_paper_started
    assert out.broker_orders_sent==0


def test_explicit_stage_is_idempotent_and_real_accounts_untouched(tmp_path,source_fixture):
    engine,work,days,now=setup(tmp_path,source_fixture)
    prior=engine.preview(clock_utc=now)
    actual=engine.stage(clock_utc=now)
    assert actual.staged_decision_id is not None
    assert actual.side==prior.side
    assert actual.quantity==prior.quantity
    assert actual.staged_decision_id==engine.stage(
        clock_utc=now+timedelta(minutes=1)
    ).staged_decision_id
    assert len(work.intents_path.read_text().splitlines())==1
    assert work.ledger.snapshot().last_sequence==0
    assert work.ledger.snapshot().cash_sek==1000


def test_no_lookahead_even_if_newer_record_exists_on_disk(tmp_path,source_fixture):
    engine,work,days,now=setup(tmp_path,source_fixture)
    with pytest.raises(ValueError,match="FUTURE_SYNTHETIC_FEED"):
        engine.preview(clock_utc=now-timedelta(days=1))
    assert not work.intents_path.exists()


def test_source_hash_race_fails_closed_before_side_effect(tmp_path,source_fixture):
    engine,work,days,now=setup(tmp_path,source_fixture)
    preview=engine.preview(clock_utc=now)
    with pytest.raises(ValueError,match="SIGNAL_SOURCE_HASH_CHANGED"):
        work.plan(
            symbol=preview.symbol, side="BUY", quantity=preview.quantity,
            clock_utc=now,
            commission_pct=engine.cfg.commission_rate,
            min_commission_sek=engine.cfg.minimum_commission_sek,
            expected_source_entry_sha256="0"*64,
        )
    with pytest.raises(ValueError,match="SIGNAL_SOURCE_SESSION_CHANGED"):
        work.plan(
            symbol=preview.symbol, side="BUY", quantity=preview.quantity,
            clock_utc=now,
            expected_source_session=date(2025,1,2),
        )
    assert not work.intents_path.exists()


def test_strategy_and_ledger_cash_must_agree(tmp_path,source_fixture):
    src,_=source_fixture
    work=SyntheticPaperWorkflow(
        feed_path=src, intents_path=tmp_path/"i", fills_path=tmp_path/"l"
    )
    with pytest.raises(ValueError,match="CASH_MISMATCH"):
        SyntheticSwingSignal(work,config=SwingConfig(initial_sek=2000))


def test_reconciled_drop_triggers_sell_before_capital_loss_gate(tmp_path,source_fixture):
    engine,work,days,now=setup(tmp_path,source_fixture)
    buy=engine.stage(clock_utc=now)
    assert buy.status=="SIGNAL"
    cal=xcals.get_calendar("XSTO")
    next_session=cal.next_session(days[-1])
    at=cal.session_close(next_session).to_pydatetime()+timedelta(minutes=20)
    # Controlled synthetic overnight loss: not a real executable fill.
    event=PaperSession(
        next_session.date(),at,"SYNTHETIC_TEST_ONLY",
        "synthetic:gap_down_test",
        # The next OPEN itself is within the precommitted risk envelope;
        # an intraday collapse after that OPEN is only visible at EOD.
        (PaperBar("AAA",buy.last_close_sek,
                  buy.last_close_sek+1.0,35.0,36.5,20000),)
    )
    work.feed.stage(event,clock_utc=at+timedelta(minutes=1))
    report=work.reconcile(clock_utc=at+timedelta(minutes=2))
    assert report[0].status=="RETROSPECTIVE_SYNTHETIC_FILL"
    assert work.ledger.snapshot().positions=={"AAA":buy.quantity}
    sell=engine.preview(clock_utc=at+timedelta(minutes=3))
    assert sell.status=="SIGNAL"
    assert sell.side=="SELL"
    assert sell.signal_reason=="CLOSE_BELOW_SMA50_TREND_EXIT"
    assert sell.quantity==buy.quantity
    staged=engine.stage(clock_utc=at+timedelta(minutes=3))
    assert staged.staged_decision_id is not None
    assert work.ledger.snapshot().last_sequence==1


def test_no_buy_if_risk_limit_or_liquidity_not_met(tmp_path,source_fixture):
    cfg=SwingConfig(min_prior_median_turnover_sek=1e15)
    engine,work,days,now=setup(tmp_path,source_fixture,cfg=cfg)
    r=engine.preview(clock_utc=now)
    assert r.status=="NO_SIGNAL"
    assert r.side is None
    assert not work.intents_path.exists()
    assert engine.stage(clock_utc=now).staged_decision_id is None

def test_stale_completed_source_does_not_offer_backdated_signal(tmp_path,source_fixture):
    engine,work,days,now=setup(tmp_path,source_fixture)
    next_open=xcals.get_calendar("XSTO").session_open(
        xcals.get_calendar("XSTO").next_session(days[-1])
    ).to_pydatetime()
    result=engine.preview(clock_utc=next_open+timedelta(seconds=1))
    assert result.status=="EXPIRED_SOURCE"
    assert result.side is None
    assert engine.stage(clock_utc=next_open+timedelta(seconds=1)).staged_decision_id is None
    assert not work.intents_path.exists()


def test_short_feed_without_205_sessions_is_not_tradable(tmp_path,source_fixture):
    engine,work,days,now=setup(tmp_path,source_fixture)
    lines=work.feed.path.read_text(encoding="utf-8").splitlines(keepends=True)
    work.feed.path.write_text("".join(lines[:204]), encoding="utf-8")
    old_date=days[203]
    c=xcals.get_calendar("XSTO")
    clock=c.session_close(old_date).to_pydatetime()+timedelta(minutes=21)
    result=engine.preview(clock_utc=clock)
    assert result.status=="INSUFFICIENT_HISTORY"
    assert engine.stage(clock_utc=clock).staged_decision_id is None
    assert not work.intents_path.exists()

def test_cli_preview_is_nonmutating_and_cannot_stage_old_sessions(tmp_path,source_fixture):
    engine,work,days,now=setup(tmp_path,source_fixture)
    script=Path(__file__).resolve().parents[2]/"scripts"/"suggest_swing_paper_signal.py"
    args=[
        sys.executable,str(script),"--feed",str(work.feed.path),
        "--intents",str(work.intents_path),
        "--ledger",str(work.ledger.path)
    ]
    preview=subprocess.run(args,capture_output=True,text=True,timeout=20,
                          cwd=script.parents[1])
    assert preview.returncode==0,preview.stderr
    report=json.loads(preview.stdout)
    assert report["status"]=="EXPIRED_SOURCE"
    assert report["staged_decision_id"] is None
    assert report["broker_orders_sent"]==0
    assert report["actual_forward_paper_started"] is False
    assert not work.intents_path.exists()
    old_stage=subprocess.run(
        args+["--stage-synthetic"],capture_output=True,text=True,
        timeout=20,cwd=script.parents[1]
    )
    assert old_stage.returncode==0,old_stage.stderr
    assert json.loads(old_stage.stdout)["status"]=="EXPIRED_SOURCE"
    assert not work.intents_path.exists()


@pytest.mark.parametrize("scenario,expected", [
    ("position_cap", "REJECTED_NEXT_OPEN_POSITION_LIMIT"),
    ("stop_risk", "REJECTED_NEXT_OPEN_STOP_RISK"),
    ("below_stop", "REJECTED_NEXT_OPEN_BELOW_PLANNED_STOP"),
])
def test_gap_open_rejections_are_deterministic_and_book_no_cash(
    tmp_path, source_fixture, scenario, expected
):
    engine, work, days, now = setup(tmp_path, source_fixture)
    candidate = engine.stage(clock_utc=now)
    assert candidate.staged_decision_id is not None
    recorded = work._replay_intents()
    assert len(recorded) == 1
    original = recorded[0]["intent"]
    assert original["max_buy_notional_sek"] == candidate.position_limit_sek
    assert original["max_buy_risk_sek"] == round(
        candidate.estimated_equity_sek * engine.cfg.risk_per_trade, 6
    )
    stop = original["reference_stop_sek"]
    shares = candidate.quantity
    if scenario == "position_cap":
        opening = original["max_buy_notional_sek"] / shares + 15
    elif scenario == "stop_risk":
        # This open must stay under the position value limit while
        # breaching the stop-loss budget inclusive of two commissions.
        risk_threshold = (
            stop + (original["max_buy_risk_sek"] -
                    2 * engine.cfg.minimum_commission_sek) / shares
        )
        opening = risk_threshold + 0.20
        assert opening * shares < original["max_buy_notional_sek"]
        assert opening > stop
    else:
        opening = stop - 1.0
    cal = xcals.get_calendar("XSTO")
    target = cal.next_session(days[-1])
    at = cal.session_close(target).to_pydatetime() + timedelta(minutes=20)
    work.feed.stage(PaperSession(
        target.date(),at,"SYNTHETIC_TEST_ONLY",
        "synthetic:next_open_risk_test",
        (PaperBar("AAA",opening,opening+1.0,opening-1.0,opening+.15,20000),)
    ),clock_utc=at+timedelta(minutes=1))
    first=work.reconcile(clock_utc=at+timedelta(minutes=2))
    assert first[0].status == expected
    assert first[0].recorded_fill_price_sek is None
    assert first[0].fee_sek is None
    assert work.ledger.snapshot().last_sequence == 0
    assert work.ledger.snapshot().cash_sek == 1000
    assert work.reconcile(clock_utc=at+timedelta(minutes=3))[0].status == expected
    # Rejected entries are terminal; they do not prevent the next
    # source session from planning a new independent research intent.
    new = work.plan(
        symbol="AAA",side="BUY",quantity=1,
        clock_utc=at+timedelta(minutes=4)
    )
    assert new != candidate.staged_decision_id
    assert len(work._replay_intents()) == 2


def test_commission_inclusive_risk_blocks_uneconomic_microposition(
    tmp_path,source_fixture
):
    # Minimum 3 SEK entry + 3 SEK exit already exceeds the 5 SEK
    # risk budget, even before an ATR price move.
    cfg=SwingConfig(minimum_commission_sek=3.0)
    engine,work,days,now=setup(tmp_path,source_fixture,cfg=cfg)
    result=engine.preview(clock_utc=now)
    assert result.status=="NO_SIGNAL"
    assert result.side is None
    assert engine.stage(clock_utc=now).staged_decision_id is None
    assert not work.intents_path.exists()


def test_envelope_is_backward_compatible_and_tamper_evident(
    tmp_path,source_fixture
):
    from stockbot.swing.paper_rehearsal import SyntheticIntent
    engine,work,days,now=setup(tmp_path,source_fixture)
    engine.stage(clock_utc=now)
    original=work._replay_intents()[0]["intent"]
    obj=SyntheticIntent.from_dict(original)
    assert obj.to_dict()==original
    assert obj.max_buy_risk_sek is not None
    text=work.intents_path.read_text()
    work.intents_path.write_text(
        text.replace('"reference_stop_sek":', '"reference_stop_sek":99.0,"unused":')
    )
    with pytest.raises(ValueError):
        work._replay_intents()
