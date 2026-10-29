"""Corporate-action JSON and CLI contract: no broker execution, strict event inputs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import exchange_calendars as xcals
import pandas as pd
import pytest

from stockbot.swing.corporate_actions_io import parse_action_records
from stockbot.swing.walkforward import WalkForwardConfig, run_walkforward
from tests.swing.test_walkforward import sample

REPO = Path(__file__).resolve().parents[2]


def _split_event(date, *, ratio=2):
    return {
        "symbol": "AAA", "ex_date": str(date.date()), "kind": "SPLIT",
        "announced_at": (date - pd.Timedelta(days=10)).strftime("%Y-%m-%dT08:00:00Z"),
        "numerator": ratio, "denominator": 1,
    }


def test_parser_rejects_unknown_fields_duplicate_schema_and_bad_dates():
    event = _split_event(pd.Timestamp("2025-12-01"))
    assert len(parse_action_records([event])) == 1
    with pytest.raises(ValueError, match="JSON list"):
        parse_action_records({"actions": [event]})
    with pytest.raises(ValueError, match="invalid action schema"):
        parse_action_records([{**event, "api_token": "must not be accepted"}])
    with pytest.raises(ValueError, match="date"):
        parse_action_records([{**event, "ex_date": "not-a-date"}])


def test_cli_accepts_synthetic_events_but_keeps_grade_unverified(tmp_path):
    dates = xcals.get_calendar("XSTO").sessions_in_range(
        "2025-01-02", "2026-03-31"
    ).tz_localize(None)[:225]
    frame = pd.DataFrame({
        "date": dates, "symbol": ["AAA"] * len(dates),
        "open": [30 + i * .08 for i in range(len(dates))],
        "high": [30.06 + i * .08 for i in range(len(dates))],
        "low": [29.9 + i * .08 for i in range(len(dates))],
        "close": [30.04 + i * .08 for i in range(len(dates))],
        "volume": [25000] * len(dates),
    })
    split_day = dates[215]
    frame.loc[215:, ["open", "high", "low", "close"]] /= 2
    csv = tmp_path / "bars.csv"
    frame.to_csv(csv, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "provider": "SYNTHETIC_ONLY",
        "terms_reference": "internal test fixture",
        "collected_at_utc": "2026-10-29T12:00:00Z",
        "price_basis": "RAW",
        "currency": "SEK"
    }), encoding="utf-8")
    events = tmp_path / "actions.json"
    events.write_text(json.dumps([_split_event(split_day)]), encoding="utf-8")
    cmd = [
        sys.executable, str(REPO / "scripts/run_swing_research.py"),
        "--csv", str(csv), "--manifest", str(manifest),
        "--actions-json", str(events),
    ]
    response = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert response.returncode == 0, response.stderr
    report = json.loads(response.stdout)
    assert report["mode"] == "BACKTEST_ONLY"
    assert report["orders_sent"] == 0
    assert report["research_grade"] == "UNVERIFIED"
    assert report["corporate_action_events"] == [f"{split_day.date()}:AAA:SPLIT"]
    denied = subprocess.run(cmd[:4] + ["--actions-json", str(events)],
                            cwd=REPO, capture_output=True, text=True)
    assert denied.returncode != 0
    assert "requires an XSTO" in denied.stderr


def test_walkforward_corporate_action_is_fold_scoped():
    df = sample(455).copy()
    split_day = pd.Timestamp(df.loc[300, "date"])
    df.loc[300:, ["open", "high", "low", "close"]] /= 2
    from stockbot.swing.corporate_actions import CorporateAction
    event = CorporateAction(
        "TEST", split_day.date(), "SPLIT",
        datetime(2023, 1, 2, tzinfo=timezone.utc),
        numerator=2, denominator=1,
    )
    config = WalkForwardConfig(train_sessions=252, test_sessions=63, min_folds=3,
                               stop_candidates=(2.0, 2.5),
                               stress_execution_bps=(10.0,))
    result = run_walkforward(df, config, corporate_actions=(event,))
    assert result.fold_count == 3
    assert result.money_traded_sek == 0.0
    assert result.data_quality == "UNVERIFIED"
