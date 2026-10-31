"""Offline lifecycle research CLI end-to-end, with synthetic cash-only delisting."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import exchange_calendars as xcals
import pandas as pd

from stockbot.swing.lifecycle import DelistingSettlement
from stockbot.swing.universe import InstrumentHistory

REPO=Path(__file__).resolve().parents[2]


def _inputs(tmp):
    days=xcals.get_calendar("XSTO").sessions_in_range(
        "2025-01-02","2026-05-01"
    ).tz_localize(None)[:265]
    rows=[]
    for sym, dates in (("AAA",days),("BBB",days[20:246])):
        for i,day in enumerate(dates):
            price=30.0 if sym=="AAA" else 20.0+i*.11
            rows.append({
                "date":day,"symbol":sym,"open":price-.04,
                "high":price+.05,"low":price-.12,
                "close":price,"volume":20000,
            })
    csv=tmp/"bars.csv"
    pd.DataFrame(rows).to_csv(csv,index=False)
    manifest=tmp/"manifest.json"
    manifest.write_text(json.dumps({
        "provider":"SYNTHETIC_ONLY",
        "terms_reference":"generated fixture, no market license",
        "collected_at_utc":"2026-10-29T22:00:00Z",
        "currency":"SEK", "price_basis":"RAW",
    }),encoding="utf-8")
    universes=[
        InstrumentHistory("AAA","SE0000000001",days[0].date(),None,
                          "2024-12-01T00:00:00Z",None,"ACTIVE"),
        InstrumentHistory("BBB","SE0000000002",days[20].date(),days[245].date(),
                          "2025-01-01T00:00:00Z","2025-01-01T00:00:00Z","DELISTED"),
    ]
    universe=tmp/"universe.json"
    universe.write_text(json.dumps([
        {**asdict(e),"first_session":str(e.first_session),
         "last_session":str(e.last_session) if e.last_session else None}
        for e in universes
    ]),encoding="utf-8")
    event=DelistingSettlement(
        "BBB",days[245].date(),2.0,days[260].date(),
        (days[245]-pd.Timedelta(days=7)).strftime("%Y-%m-%dT07:00:00Z"),
        "SYNTHETIC_CASH_ONLY_RECEIPT",
    )
    settlement=tmp/"settlements.json"
    settlement.write_text(json.dumps([
        {**asdict(event),"last_session":str(event.last_session),
         "pay_date":str(event.pay_date)}
    ]),encoding="utf-8")
    return csv,manifest,universe,settlement


def run(*args):
    return subprocess.run(
        [sys.executable,str(REPO/"scripts/run_swing_research.py"),
         *(str(x) for x in args)],
        cwd=REPO,capture_output=True,text=True,timeout=60,
    )


def test_dynamic_universe_cli_cash_delist_without_real_orders(tmp_path):
    csv,manifest,universe,settlement=_inputs(tmp_path)
    p=run("--csv",csv,"--manifest",manifest,"--universe-json",universe,
          "--settlements-json",settlement)
    assert p.returncode==0,p.stderr
    report=json.loads(p.stdout)
    assert report["mode"]=="BACKTEST_ONLY"
    assert report["orders_sent"]==0
    assert report["data_quality"]["status"]=="STRUCTURAL_LIFECYCLE_ONLY_UNVERIFIED"
    assert report["universe_audit"]["compatible_with_current_fixed_panel_engine"] is False
    assert report["delisted_shares"] > 0
    assert report["delist_cash_paid_sek"] > 0
    assert report["delist_receivable_sek"] == 0
    assert len(report["lifecycle_events"]) == 1


def test_dynamic_walkforward_with_too_few_sessions_blocks_without_phantom_trades(tmp_path):
    csv,manifest,universe,settlement=_inputs(tmp_path)
    p=run("--csv",csv,"--manifest",manifest,"--universe-json",universe,
          "--settlements-json",settlement,"--walkforward")
    assert p.returncode!=0
    assert "insufficient complete sessions for OOS" in p.stderr


def test_settlement_file_cannot_be_submitted_without_universe(tmp_path):
    csv,manifest,universe,settlement=_inputs(tmp_path)
    p=run("--csv",csv,"--manifest",manifest,
          "--settlements-json",settlement)
    assert p.returncode!=0
    assert "requires --universe-json" in p.stderr


def test_lifecycle_cli_rejects_adjusted_price_basis_before_simulating(tmp_path):
    csv,manifest,universe,settlement=_inputs(tmp_path)
    raw=json.loads(manifest.read_text(encoding="utf-8"))
    raw["price_basis"]="SPLIT_ADJUSTED"
    manifest.write_text(json.dumps(raw),encoding="utf-8")
    p=run("--csv",csv,"--manifest",manifest,"--universe-json",universe,
          "--settlements-json",settlement)
    assert p.returncode!=0
    assert "requires price_basis=RAW" in p.stderr
