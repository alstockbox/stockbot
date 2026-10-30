"""Reference-data CLI smoke tests: synthetic metadata, no market downloads."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from stockbot.swing.universe import InstrumentHistory

REPO = Path(__file__).resolve().parents[2]


def _fixtures(tmp: Path):
    import exchange_calendars as xcals
    days = xcals.get_calendar("XSTO").sessions_in_range(
        "2025-01-02", "2026-01-30"
    ).tz_localize(None)[:220]
    rows = []
    for symbol, dates in (("ACTIVE", days), ("GONE", days[50:170])):
        for i, day in enumerate(dates):
            px = 30 + i*.05
            rows.append({"date":day, "symbol":symbol,
                         "open":px, "high":px+.2, "low":px-.2,
                         "close":px, "volume":30000})
    csv = tmp / "bars.csv"
    pd.DataFrame(rows).to_csv(csv,index=False)
    records = [
        InstrumentHistory(
            "ACTIVE", "SE0000000001", days[0].date(), None,
            "2024-12-01T00:00:00Z", None, "ACTIVE",
        ),
        InstrumentHistory(
            "GONE", "SE0000000002", days[50].date(), days[169].date(),
            "2025-01-02T00:00:00Z", "2025-06-01T00:00:00Z", "DELISTED",
        ),
    ]
    universe = tmp / "universe.json"
    universe.write_text(json.dumps([
        {**asdict(r), "first_session": str(r.first_session),
         "last_session": str(r.last_session) if r.last_session else None}
        for r in records
    ]), encoding="utf-8")
    manifest = tmp / "provenance.json"
    manifest.write_text(json.dumps({
        "provider":"INTERNAL_SYNTHETIC_ONLY",
        "terms_reference":"fixture, no exchange rights claimed",
        "collected_at_utc":"2026-10-29T22:00:00Z",
        "currency":"SEK","price_basis":"RAW",
    }), encoding="utf-8")
    return csv, universe, manifest


def _run(script, *args):
    return subprocess.run(
        [sys.executable, str(REPO/"scripts"/script),
         *(str(a) for a in args)],
        cwd=REPO, capture_output=True, text=True,
    )


def test_offline_universe_cli_accepts_irregular_history_for_audit_only(tmp_path):
    csv, universe, manifest = _fixtures(tmp_path)
    p = _run("audit_swing_universe.py", "--csv", csv,
             "--universe-json", universe, "--manifest", manifest)
    assert p.returncode == 0, p.stderr
    receipt = json.loads(p.stdout)
    assert receipt["mode"] == "STRUCTURAL_UNIVERSE_AUDIT_ONLY"
    assert receipt["compatible_with_current_fixed_panel_engine"] is False
    assert receipt["order_execution_permitted"] is False
    assert receipt["historical_universe_independently_verified"] is False


def test_trading_research_cli_rejects_uneven_lifecycles(tmp_path):
    csv, universe, manifest = _fixtures(tmp_path)
    p = _run("run_swing_research.py", "--csv", csv,
             "--universe-json", universe, "--manifest", manifest)
    assert p.returncode != 0
    assert "missing or unexpected delisting cash settlement" in p.stderr


def test_cli_refuses_unsourced_benchmark_or_market_data_without_audit(tmp_path):
    csv, universe, manifest = _fixtures(tmp_path)
    p = _run("run_swing_research.py", "--csv", csv,
             "--benchmark-csv", csv)
    assert p.returncode != 0
    assert "provided together" in p.stderr
    p = _run("run_swing_research.py", "--csv", csv,
             "--universe-json", universe)
    assert p.returncode != 0
    assert "requires --manifest" in p.stderr
