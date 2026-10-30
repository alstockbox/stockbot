"""No-network/no-broker synthetic XSTO feed CLI acceptance tests."""
from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys

import exchange_calendars as xcals
import pandas as pd

from stockbot.swing.paper_feed import PaperBar, PaperSession

REPO=Path(__file__).resolve().parents[2]


def execute(*args):
    return subprocess.run(
        [sys.executable,str(REPO/"scripts"/"stage_swing_paper_feed.py"),
         *(str(x) for x in args)],
        cwd=REPO,text=True,capture_output=True,timeout=20
    )


def test_cli_stages_once_and_blocks_replay_and_third_party_feed(tmp_path):
    calendar=xcals.get_calendar("XSTO")
    observed=calendar.session_close(pd.Timestamp("2025-01-02")).to_pydatetime()+timedelta(minutes=20)
    event=PaperSession(
        pd.Timestamp("2025-01-02").date(),observed,
        "SYNTHETIC_TEST_ONLY","synthetic:cli_test",
        (PaperBar("AAA",10,12,9,11,10_000),)
    )
    source=tmp_path/"session.json"
    source.write_text(json.dumps(event.to_dict()),encoding="utf-8")
    journal=tmp_path/"feed.jsonl"
    first=execute("--journal",journal,"--session-json",source)
    assert first.returncode==0,first.stderr
    report=json.loads(first.stdout)
    assert report["mode"]=="SYNTHETIC_PAPER_REHEARSAL_ONLY"
    assert report["sessions_staged"]==1
    assert report["broker_orders_sent"]==0
    assert report["readiness"]["real_forward_paper_allowed"] is False
    second=execute("--journal",journal)
    assert second.returncode==0,second.stderr
    assert json.loads(second.stdout)["chain_head_sha256"]==report["chain_head_sha256"]
    duplicate=execute("--journal",journal,"--session-json",source)
    assert duplicate.returncode!=0
    assert "DUPLICATE_OR_OUT_OF_ORDER_SESSION" in duplicate.stderr
    external=event.to_dict()
    external["source_kind"]="NASDAQ_DELAYED_15MIN"
    source.write_text(json.dumps(external),encoding="utf-8")
    not_licensed=execute("--journal",journal,"--session-json",source)
    assert not_licensed.returncode!=0
    assert "EXTERNAL_MARKET_DATA_UNLICENSED_OR_UNVERIFIED" in not_licensed.stderr
    assert json.loads(execute("--journal",journal).stdout)["sessions_staged"]==1
