"""E2E CLI: only reconstructs synthetic past sessions, never orders."""
from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys

from tests.swing.test_paper_rehearsal import env, stage, dates

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/"scripts"/"rehearse_swing_paper.py"


def cli(work, *extra):
    return subprocess.run(
        [sys.executable,str(SCRIPT),
         "--feed",str(work.feed.path),
         "--intents",str(work.intents_path),
         "--ledger",str(work.ledger.path),*extra],
        capture_output=True,text=True,timeout=20,cwd=ROOT,
    )


def test_cli_reconstructs_one_hypothetical_fill_without_account_access(tmp_path):
    work,feed,ledger=env(tmp_path)
    first,second,_=dates()
    a=stage(feed,first,100)
    intent=work.plan(symbol="AAA",side="BUY",quantity=1,
                     clock_utc=a.observed_at_utc+timedelta(minutes=1))
    b=stage(feed,second,104)
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
    work,feed,ledger=env(tmp_path)
    stage(feed,dates()[0])
    result=cli(work,"--plan","--symbol","AAA","--side","BUY","--quantity","1")
    assert result.returncode!=0
    assert "decision created after target market opened" in result.stderr
    assert not work.intents_path.exists()
    assert ledger.snapshot().last_sequence==0


def test_cli_rejects_missing_plan_details(tmp_path):
    work,feed,ledger=env(tmp_path)
    result=cli(work,"--plan","--symbol","AAA")
    assert result.returncode!=0
    assert "--plan requires" in result.stderr
