"""End-to-end CLI smoke checks: only local synthetic data, no live broker."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import exchange_calendars as xcals
import pandas as pd


REPO = Path(__file__).resolve().parents[2]


def _make_inputs(tmp_path: Path):
    dates = xcals.get_calendar("XSTO").sessions_in_range(
        "2025-01-02", "2026-05-01"
    ).tz_localize(None)[:225]
    n = len(dates)
    frame = pd.DataFrame({
        "date": dates, "symbol": ["TEST.ST"] * n,
        "open": [100 + i * 0.06 for i in range(n)],
        "high": [101 + i * 0.06 for i in range(n)],
        "low": [99 + i * 0.06 for i in range(n)],
        "close": [100.4 + i * 0.06 for i in range(n)],
        "volume": [100000] * n,
    })
    csv_path = tmp_path / "synthetic.csv"
    frame.to_csv(csv_path, index=False)
    manifest_path = tmp_path / "provenance.json"
    manifest_path.write_text(json.dumps({
        "provider": "SYNTHETIC_TEST_ONLY",
        "terms_reference": "internal deterministic fixture; no external provider",
        "collected_at_utc": "2026-10-29T12:00:00Z",
        "currency": "SEK",
        "price_basis": "RAW",
        "corporate_actions_checked": False,
        "historical_universe_checked": False,
    }), encoding="utf-8")
    return frame, csv_path, manifest_path


def _run(csv: Path, manifest: Path, output: Path):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts/run_swing_research.py"),
         "--csv", str(csv), "--manifest", str(manifest),
         "--output", str(output)],
        cwd=REPO, capture_output=True, text=True, check=False,
    )


def test_cli_xsto_receipt_and_no_order(tmp_path):
    _, csv, manifest = _make_inputs(tmp_path)
    outfile = tmp_path / "report.json"
    completed = _run(csv, manifest, outfile)
    assert completed.returncode == 0, completed.stderr
    output = json.loads(outfile.read_text(encoding="utf-8"))
    assert output["mode"] == "BACKTEST_ONLY"
    assert output["orders_sent"] == 0
    assert output["research_grade"] == "UNVERIFIED"
    assert output["data_quality"]["exchange"] == "XSTO"
    assert output["data_quality"]["order_execution_permitted"] is False
    assert output["data_quality"]["bars"] == 225


def test_cli_fails_before_backtest_report_on_invalid_session(tmp_path):
    frame, csv, manifest = _make_inputs(tmp_path)
    frame.loc[5, "date"] = "2025-01-01"
    frame.to_csv(csv, index=False)
    outfile = tmp_path / "must_not_exist.json"
    completed = _run(csv, manifest, outfile)
    assert completed.returncode != 0
    assert not outfile.exists()
    assert "non-XSTO session" in completed.stderr
