"""End-to-end CLI benchmark/hedged universe labels. Synthetic input only."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import exchange_calendars as xcals
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def test_cli_walkforward_gi_comparison_never_claims_licensed_alpha(tmp_path):
    sessions = xcals.get_calendar("XSTO").sessions_in_range(
        "2022-01-03", "2026-01-02"
    ).tz_localize(None)[:900]
    assert len(sessions) == 900
    bars = pd.DataFrame({
        "date":sessions, "symbol":["AAA"]*len(sessions),
        "open":[30 + i*.03 for i in range(len(sessions))],
        "high":[30.06 + i*.03 for i in range(len(sessions))],
        "low":[29.94 + i*.03 for i in range(len(sessions))],
        "close":[30.02 + i*.03 for i in range(len(sessions))],
        "volume":[10000]*len(sessions),
    })
    stock_path = tmp_path/"stocks.csv"
    bars.to_csv(stock_path,index=False)
    provenance = tmp_path/"source.json"
    provenance.write_text(json.dumps({
        "provider":"SYNTHETIC_FIXTURE_ONLY",
        "terms_reference":"internal generated data",
        "collected_at_utc":"2026-10-29T22:00:00Z",
        "currency":"SEK","price_basis":"RAW",
    }),encoding="utf-8")
    index_file = tmp_path/"benchmark.csv"
    pd.DataFrame({
        "date":sessions,
        "close":[100 + i*.02 for i in range(len(sessions))]
    }).to_csv(index_file,index=False)
    index_manifest = tmp_path/"benchmark_source.json"
    index_manifest.write_text(json.dumps({
        "index_code":"OMXSBGI",
        "source":"SYNTHETIC_FIXTURE_ONLY",
        "usage_rights_reference":"internal generated data",
        "collected_at_utc":"2026-10-29T22:00:00Z",
        "currency":"SEK","return_basis":"GROSS_TOTAL_RETURN",
    }),encoding="utf-8")
    p = subprocess.run([
        sys.executable,str(REPO/"scripts/run_swing_research.py"),
        "--csv",str(stock_path),"--manifest",str(provenance),
        "--walkforward","--benchmark-csv",str(index_file),
        "--benchmark-manifest",str(index_manifest),
    ],cwd=REPO,capture_output=True,text=True,timeout=110)
    assert p.returncode == 0,p.stderr
    report=json.loads(p.stdout)
    assert report["mode"]=="WALK_FORWARD_BACKTEST_ONLY"
    assert report["orders_sent"] == 0
    assert report["universe_audit"]["status"]=="NOT_AUDITED"
    assert report["fold_count"] >= 3
    assert report["benchmark_comparison"]["mode"]=="UNVERIFIED_OOS_BENCHMARK_COMPARISON"
    assert report["benchmark_comparison"]["alpha_claim_permitted"] is False
    assert report["benchmark_comparison"]["license_independently_verified"] is False
