"""Synthetic IPO/delisting lifecycle audit tests; no real market-data assertions."""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import exchange_calendars as xcals
import pandas as pd
import pytest

from stockbot.swing.universe import (
    InstrumentHistory, audit_instrument_history, parse_universe_records,
)


def fixture():
    days = xcals.get_calendar("XSTO").sessions_in_range(
        "2025-01-02", "2026-01-30"
    ).tz_localize(None)[:220]
    rows = []
    def add(symbol, dates):
        for i, d in enumerate(dates):
            p = 80 + i*.07
            rows.append({"date":d, "symbol":symbol,
                         "open":p, "high":p+1, "low":p-1,
                         "close":p+.1, "volume":10000})
    add("AAA", days)
    add("BBB", days[40:160])
    records = (
        InstrumentHistory("AAA", "SE0000000001", days[0].date(), None,
                          "2024-12-01T00:00:00Z", None, "ACTIVE"),
        InstrumentHistory("BBB", "SE0000000002", days[40].date(), days[159].date(),
                          "2025-01-01T00:00:00Z", "2025-06-01T00:00:00Z",
                          "DELISTED"),
    )
    return pd.DataFrame(rows), days, records


def test_audit_accepts_uneven_actual_lifecycles_but_fails_open_for_replay():
    data, days, records = fixture()
    report = audit_instrument_history(
        data, records, collected_at_utc="2026-10-29T22:00:00Z"
    )
    assert report.mode == "STRUCTURAL_UNIVERSE_AUDIT_ONLY"
    assert report.symbols_in_bars == 2
    assert report.delisted_symbols_in_bars == 1
    assert report.listing_transitions_inside_window == 1
    assert report.delisting_transitions_inside_window == 1
    assert report.compatible_with_current_fixed_panel_engine is False
    assert report.historical_universe_independently_verified is False
    assert report.authorized_for_paper_or_live is False
    assert len(report.data_sha256) == 64


def test_audit_rejects_survivor_only_bar_set_when_delisted_member_required():
    data, days, records = fixture()
    survivors = data[data["symbol"] == "AAA"]
    with pytest.raises(ValueError, match="missing eligible instrument"):
        audit_instrument_history(
            survivors, records, collected_at_utc="2026-10-29T22:00:00Z"
        )


def test_audit_rejects_missing_mid_listing_and_phantom_after_delist():
    data, days, records = fixture()
    missing = data.loc[~((data["symbol"] == "BBB") &
                         (data["date"] == days[120]))]
    with pytest.raises(ValueError, match="missing tradable-session"):
        audit_instrument_history(
            missing, records, collected_at_utc="2026-10-29T22:00:00Z"
        )
    ghost = data.copy()
    ghost = pd.concat([ghost, data.loc[
        (data["symbol"] == "BBB") & (data["date"] == days[159])
    ].assign(date=days[160])], ignore_index=True)
    with pytest.raises(ValueError, match="phantom bar"):
        audit_instrument_history(
            ghost, records, collected_at_utc="2026-10-29T22:00:00Z"
        )


def test_audit_rejects_unknown_symbol_and_bogus_time():
    data, days, records = fixture()
    with pytest.raises(ValueError, match="absent from historical universe"):
        audit_instrument_history(
            data, records[:1], collected_at_utc="2026-10-29T22:00:00Z"
        )
    with pytest.raises(ValueError, match="collected before close"):
        audit_instrument_history(
            data, records, collected_at_utc="2025-01-10T09:00:00Z"
        )


def test_universe_json_strict_schema_and_no_duplicate_isins():
    _, days, records = fixture()
    import dataclasses
    raw = []
    for r in records:
        raw.append({**dataclasses.asdict(r),
                    "first_session":str(r.first_session),
                    "last_session":str(r.last_session) if r.last_session else None})
    assert parse_universe_records(raw) == records
    with pytest.raises(ValueError, match="unknown"):
        parse_universe_records([{**raw[0], "test_field": True}])
    duplicate = {**raw[1], "isin":raw[0]["isin"]}
    with pytest.raises(ValueError, match="ambiguous ISIN"):
        parse_universe_records([raw[0], duplicate])


def test_universe_listing_cannot_be_known_after_open():
    _, days, records = fixture()
    with pytest.raises(ValueError, match="before first open"):
        replace(records[1], first_known_at_utc="2026-01-01T00:00:00Z")


def test_complete_fixed_panel_is_structurally_compatible_not_verified():
    data, days, records = fixture()
    only_aaa = data[data["symbol"] == "AAA"]
    report = audit_instrument_history(only_aaa, records[:1],
                                     collected_at_utc="2026-10-29T22:00:00Z")
    assert report.compatible_with_current_fixed_panel_engine is True
    assert report.historical_universe_independently_verified is False

def test_future_disclosure_not_retroactively_known_in_data_snapshot():
    data, days, records = fixture()
    edited = replace(records[1], last_known_at_utc="2027-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="after collection as-of"):
        audit_instrument_history(
            data, (records[0], edited),
            collected_at_utc="2026-10-29T22:00:00Z"
        )
