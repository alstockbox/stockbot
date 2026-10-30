"""Dynamic point-in-time XSTO universe walk-forward. Synthetic only.

Proves IPOs and cash-only delistings do not leak into earlier fold decisions.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
import exchange_calendars as xcals
import pandas as pd
import pytest

from stockbot.swing.lifecycle import DelistingSettlement
from stockbot.swing.universe import InstrumentHistory
from stockbot.swing.walkforward import WalkForwardConfig, run_walkforward


ASOF = "2026-10-29T22:00:00Z"


def synthetic_lifecycle():
    days = xcals.get_calendar("XSTO").sessions_in_range(
        "2025-01-02", "2027-01-01"
    ).tz_localize(None)[:455]
    assert len(days) == 455
    rows = []
    for i, day in enumerate(days):
        a = 25.0
        rows.append({"date": day, "symbol":"AAA", "open":a-.03,
                     "high":a+.05,"low":a-.1,"close":a,"volume":25000})
        if 40 <= i <= 340:
            p = 20.0 + (i-40)*.11
            rows.append({"date": day,"symbol":"BBB",
                         "open":p-.04,"high":p+.05,"low":p-.12,
                         "close":p,"volume":40000})
    universe = (
        InstrumentHistory("AAA","SE0000000001",days[0].date(),None,
                          "2024-12-01T00:00:00Z",None,"ACTIVE"),
        InstrumentHistory("BBB","SE0000000002",days[40].date(),
                          days[340].date(),"2025-01-01T00:00:00Z",
                          "2025-01-01T00:00:00Z","DELISTED"),
    )
    event = DelistingSettlement(
        "BBB", days[340].date(), 2.0, days[350].date(),
        "2025-01-01T00:00:00Z", "SYNTHETIC_ONLY",
    )
    return pd.DataFrame(rows), days, universe, event


def config():
    return WalkForwardConfig(
        train_sessions=252, test_sessions=63, min_folds=3,
        stop_candidates=(2.5,), stress_execution_bps=(15.0,),
    )


def test_ipo_and_cash_delist_cross_fold_oos_without_lookahead():
    data, days, records, settlement = synthetic_lifecycle()
    result = run_walkforward(
        data, config(), universe=records, settlements=(settlement,),
        collected_at_utc=ASOF,
    )
    assert result.fold_count == 3
    assert result.mode == "WALK_FORWARD_BACKTEST_ONLY"
    assert result.source_universe_sha256 is not None
    assert result.historical_universe_independently_verified is False
    assert result.money_traded_sek == 0
    assert result.sufficient_activity_for_live_decision is False
    assert [f.train_start for f in result.folds] == [
        str(days[i].date()) for i in (0,63,126)
    ]
    assert result.folds[0].test_universe_symbols == 2
    assert result.folds[1].test_universe_symbols == 2
    assert result.folds[2].train_universe_symbols == 2
    assert result.folds[0].test_lifecycle_events == ()
    assert result.folds[1].test_lifecycle_events == (
        f"{days[340].date()}:BBB:DELISTED_CASH_ONLY",
    )
    assert result.folds[2].test_lifecycle_events == ()


def test_future_delisting_cash_amount_never_changes_first_fold():
    data, days, records, settlement = synthetic_lifecycle()
    first = run_walkforward(
        data, config(), universe=records, settlements=(settlement,),
        collected_at_utc=ASOF,
    )
    # Change the known settlement *far after* fold one's test end (day 314).
    second = run_walkforward(
        data, config(), universe=records,
        settlements=(replace(settlement, cash_per_share_sek=0),),
        collected_at_utc=ASOF,
    )
    assert first.folds[0] == second.folds[0]
    assert first.source_settlements_sha256 != second.source_settlements_sha256
    assert first.data_sha256 == second.data_sha256
    assert first.source_universe_sha256 == second.source_universe_sha256
    assert first.folds[1].chosen_atr_stop_multiple == second.folds[1].chosen_atr_stop_multiple
    assert first.folds[1].training_net_return_pct == second.folds[1].training_net_return_pct


def test_incomplete_delisting_settlement_fail_closed():
    data, days, records, settlement = synthetic_lifecycle()
    with pytest.raises(ValueError,match="missing or unexpected delisting"):
        run_walkforward(data, config(), universe=records,
                        collected_at_utc=ASOF)
    with pytest.raises(ValueError,match="historical universe"):
        run_walkforward(data,config(),settlements=(settlement,))
    with pytest.raises(ValueError,match="collection as-of"):
        run_walkforward(data,config(),universe=records,settlements=(settlement,))


def test_cli_can_run_synthetic_dynamic_oos_without_orders(tmp_path, monkeypatch):
    """Exercise actual CLI path with short, explicit synthetic OOS config."""
    frame, days, records, settlement = synthetic_lifecycle()
    csv = tmp_path / "bars.csv"
    frame.to_csv(csv, index=False)
    meta = tmp_path / "manifest.json"
    meta.write_text(json.dumps({
        "provider": "SYNTHETIC_ONLY",
        "terms_reference": "generated internal research fixture",
        "collected_at_utc": ASOF,
        "currency": "SEK",
        "price_basis": "RAW",
    }), encoding="utf-8")
    history = tmp_path / "universe.json"
    history.write_text(json.dumps([asdict(r) for r in records], default=str),
                       encoding="utf-8")
    settlements = tmp_path / "settlements.json"
    settlements.write_text(json.dumps([asdict(settlement)], default=str),
                           encoding="utf-8")
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_swing_research.py"
    spec = importlib.util.spec_from_file_location("dynamic_swing_cli_fixture", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Keep the production CLI defaults (504/126) unchanged; the test only
    # reduces required synthetic bars via monkeypatched test configuration.
    monkeypatch.setattr(
        module, "WalkForwardConfig",
        lambda base: replace(config(), base=base),
    )
    report_path = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", [
        str(script), "--csv", str(csv), "--manifest", str(meta),
        "--universe-json", str(history),
        "--settlements-json", str(settlements), "--walkforward",
        "--output", str(report_path),
    ])
    assert module.main() == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["orders_sent"] == 0
    assert report["mode"] == "WALK_FORWARD_BACKTEST_ONLY"
    assert report["fold_count"] == 3
    assert report["historical_universe_independently_verified"] is False
    assert report["data_quality"]["license_independently_verified"] is False
    assert report["data_quality"]["order_execution_permitted"] is False
    assert report["source_universe_sha256"] is not None
    assert report["source_settlements_sha256"] is not None
    assert report["folds"][0]["test_lifecycle_events"] == []
    assert report["folds"][1]["test_lifecycle_events"] == [
        f"{days[340].date()}:BBB:DELISTED_CASH_ONLY"
    ]
