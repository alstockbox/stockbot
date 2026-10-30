"""Self-contained E2E CLI tests, compatible with isolated pytest/Linux import."""
from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys

import exchange_calendars as xcals
import pandas as pd

from stockbot.swing.paper_feed import PaperBar, PaperSession, PaperFeedJournal
from stockbot.swing.paper_ledger import SEKPaperLedger
from stockbot.swing.paper_rehearsal import SyntheticPaperWorkflow

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/"scripts"/"rehearse_swing_paper.py"


def fixture(tmp_path):
    feed_path=tmp_path/"feed.jsonl"
    workflow=SyntheticPaperWorkflow(
        feed_path=feed_path,
        intents_path=tmp_path/"intents.jsonl",
        fills_path=tmp_path/"fills.jsonl",
    )
    return workflow, PaperFeedJournal(feed_path), SEKPaperLedger(tmp_path/"fills.jsonl")


def trading_days():
    cal=xcals.get_calendar("XSTO")
    first=pd.Timestamp("2025-01-02")
    second=cal.next_session(first)
    return first.date(),second.date()


def stage(feed, day, price=100):
    closed=xcals.get_calendar("XSTO").session_close(pd.Timestamp(day)).to_pydatetime()
    observed=closed+timedelta(minutes=20)
    event=PaperSession(
        session=day, observed_at_utc=observed,
        source_kind="SYNTHETIC_TEST_ONLY",
        source_reference="synthetic:isolated_ci_fixture",
        bars=(PaperBar("AAA",price,price+2,price-2,price+1,10000),),
    )
    feed.stage(event,clock_utc=observed+timedelta(minutes=1))
    return event


def cli(work, *extra):
    return subprocess.run(
        [sys.executable,str(SCRIPT),
         "--feed",str(work.feed.path),
         "--intents",str(work.intents_path),
         "--ledger",str(work.ledger.path),*extra],
        capture_output=True,text=True,timeout=20,cwd=ROOT,
    )


def test_cli_reconstructs_one_hypothetical_fill_without_account_access(tmp_path):
    work,feed,ledger=fixture(tmp_path)
    first,second=trading_days()
    a=stage(feed,first,100)
    intent=work.plan(symbol="AAA",side="BUY",quantity=1,
                     clock_utc=a.observed_at_utc+timedelta(minutes=1))
    stage(feed,second,104)
    output=cli(work,"--reconcile")
    assert output.returncode==0,output.stderr
    result=json.loads(output.stdout)
    assert result["mode"]=="SYNTHETIC_RETROSPECTIVE_NEXT_XSTO_OPEN"
    assert result["real_forward_paper_started"] is False
    assert result["broker_orders_sent"]==0
    assert len(result["reconciled"])==1
    assert result["reconciled"][0]["decision_id"]==intent
    assert result["reconciled"][0]["recorded_fill_price_sek"]==104
    assert result["reconciled"][0]["status"]=="RETROSPECTIVE_SYNTHETIC_FILL"
    repeated=cli(work,"--reconcile")
    assert repeated.returncode==0,repeated.stderr
    assert json.loads(repeated.stdout)["reconciled"][0]["status"]=="RECONSTRUCTED_PREVIOUSLY"
    assert ledger.snapshot().last_sequence==1


def test_cli_cannot_generate_historically_backdated_new_signal(tmp_path):
    work,feed,ledger=fixture(tmp_path)
    stage(feed,trading_days()[0])
    result=cli(work,"--plan","--symbol","AAA","--side","BUY","--quantity","1")
    assert result.returncode!=0
    assert "decision created after target market opened" in result.stderr
    assert not work.intents_path.exists()
    assert ledger.snapshot().last_sequence==0


def test_cli_rejects_missing_plan_details(tmp_path):
    work,feed,ledger=fixture(tmp_path)
    result=cli(work,"--plan","--symbol","AAA")
    assert result.returncode!=0
    assert "--plan requires" in result.stderr
