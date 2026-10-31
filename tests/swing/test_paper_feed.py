"""Synthetic-only paper feed integrity / as-of / provenance tests."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json

import exchange_calendars as xcals
import pandas as pd
import pytest

from stockbot.swing.paper_feed import (
    GENESIS, PaperBar, PaperFeedJournal, PaperSession,
)


def observed(day):
    close=xcals.get_calendar("XSTO").session_close(pd.Timestamp(day)).to_pydatetime()
    return close+timedelta(minutes=20)


def bar(symbol="AAA", **overrides):
    data={"symbol":symbol,"open":100.0,"high":103.0,"low":98.0,
          "close":101.0,"volume":10000}
    data.update(overrides)
    return PaperBar(**data)


def record(day="2025-01-02", *, symbols=("AAA",), **overrides):
    data={
        "session":pd.Timestamp(day).date(),
        "observed_at_utc":observed(day),
        "source_kind":"SYNTHETIC_TEST_ONLY",
        "source_reference":"synthetic:test_data",
        "bars":tuple(bar(s) for s in symbols),
    }
    data.update(overrides)
    return PaperSession(**data)


def test_replay_persists_two_sessions_and_stays_nonlive(tmp_path):
    j=PaperFeedJournal(tmp_path/"paper_sessions.jsonl")
    initial=j.snapshot()
    assert initial.sessions_staged==0 and initial.chain_head_sha256==GENESIS
    day1=record("2025-01-02",symbols=("BBB","AAA"))
    s1=j.stage(day1,clock_utc=observed("2025-01-02")+timedelta(minutes=1))
    assert s1.sessions_staged==1
    assert s1.symbols_on_last_session==2
    assert not s1.real_forward_paper_started
    assert not s1.actual_market_data_rights_verified
    assert s1.broker_orders_sent==0
    day2=record("2025-01-03")
    s2=j.stage(day2,clock_utc=observed("2025-01-03")+timedelta(hours=1))
    assert s2.sessions_staged==2
    assert s2.chain_head_sha256!=s1.chain_head_sha256
    assert len(s2.chain_head_sha256)==64
    assert PaperFeedJournal(j.path).snapshot()==s2


def test_duplicate_and_backward_session_fail_closed(tmp_path):
    j=PaperFeedJournal(tmp_path/"feed")
    jan3=record("2025-01-03")
    j.stage(jan3,clock_utc=observed("2025-01-03")+timedelta(hours=1))
    with pytest.raises(ValueError,match="DUPLICATE_OR_OUT_OF_ORDER_SESSION"):
        j.stage(jan3,clock_utc=observed("2025-01-03")+timedelta(hours=1))
    with pytest.raises(ValueError,match="DUPLICATE_OR_OUT_OF_ORDER_SESSION"):
        j.stage(record("2025-01-02"),
                clock_utc=observed("2025-01-03")+timedelta(hours=1))
    assert j.snapshot().sessions_staged==1


def test_prior_and_current_day_gaps_explicit_not_backfilled(tmp_path):
    j=PaperFeedJournal(tmp_path/"feed")
    j.stage(record("2025-01-02"),
            clock_utc=observed("2025-01-02")+timedelta(hours=1))
    j.stage(record("2025-01-07"),
            clock_utc=observed("2025-01-07")+timedelta(hours=1))
    # The journal is a raw staging queue; a separate readiness gate must
    # refuse to run strategies on incomplete calendar history.
    assert j.snapshot().sessions_staged==2


def test_tampering_hash_and_duplicate_sequence_detected(tmp_path):
    path=tmp_path/"feed.jsonl"
    j=PaperFeedJournal(path)
    j.stage(record(),clock_utc=observed("2025-01-02")+timedelta(hours=1))
    text=path.read_text()
    path.write_text(text.replace('"close":101.0','"close":102.0'))
    with pytest.raises(ValueError,match="tamper/hash"):
        j.snapshot()
    with pytest.raises(ValueError,match="tamper/hash"):
        j.stage(record("2025-01-03"),
                clock_utc=observed("2025-01-03")+timedelta(hours=1))


def test_no_early_bar_or_future_observation(tmp_path):
    close=xcals.get_calendar("XSTO").session_close(pd.Timestamp("2025-01-02")).to_pydatetime()
    with pytest.raises(ValueError,match="not available"):
        record(observed_at_utc=close)
    j=PaperFeedJournal(tmp_path/"feed")
    with pytest.raises(ValueError,match="in the future"):
        j.stage(record(),clock_utc=close)


def test_external_market_data_including_delayed_nasdaq_refused():
    with pytest.raises(ValueError,match="EXTERNAL_MARKET_DATA"):
        record(source_kind="NASDAQ_DELAYED_15MIN")
    with pytest.raises(ValueError,match="synthetic:"):
        record(source_reference="https://tradereports.nasdaq.com")


def test_invalid_bars_dates_and_symbols_refused():
    with pytest.raises(ValueError,match="non-XSTO"):
        PaperSession(pd.Timestamp("2025-01-04").date(),
                     datetime(2025,1,5,tzinfo=timezone.utc),
                     "SYNTHETIC_TEST_ONLY","synthetic:test",(bar(),))
    with pytest.raises(ValueError,match="duplicate"):
        record(symbols=("AAA","AAA"))
    with pytest.raises(ValueError,match="positive finite"):
        bar(close=float("nan"))
    with pytest.raises(ValueError,match="inconsistent"):
        bar(high=80)
    with pytest.raises(ValueError,match="nonnegative integer"):
        bar(volume=4.5)
    with pytest.raises(ValueError,match="invalid paper symbol"):
        bar("AA A")


def test_json_schema_fails_closed_on_unknown_fields_and_ambiguous_clock():
    good=record().to_dict()
    assert PaperSession.from_dict(good).to_dict()==good
    with pytest.raises(ValueError,match="schema"):
        PaperSession.from_dict({**good,"leak":"broker-key"})
    with pytest.raises(ValueError,match="schema"):
        PaperSession.from_dict({**good,"bars":[{**good["bars"][0],"new":1}]})
    with pytest.raises(ValueError,match="UTC"):
        PaperSession.from_dict({**good,"observed_at_utc":"2025-01-02T17:00:00"})
    with pytest.raises(ValueError,match="timezone-aware"):
        PaperFeedJournal("unused").stage(
            record(),clock_utc=datetime(2026,1,1)
        )


def test_canonical_bar_order_is_stable():
    a=record(symbols=("BBB","AAA"))
    b=record(symbols=("AAA","BBB"))
    assert a.to_dict()==b.to_dict()

def test_readiness_separates_gaps_from_legal_permission(tmp_path):
    j=PaperFeedJournal(tmp_path/"feed")
    j.stage(record("2025-01-02"),
            clock_utc=observed("2025-01-02")+timedelta(hours=1))
    j.stage(record("2025-01-07"),
            clock_utc=observed("2025-01-07")+timedelta(hours=1))
    report=j.readiness()
    assert report.sessions_staged==2
    assert not report.complete_xsto_calendar
    assert report.constant_symbol_universe
    assert not report.technical_warmup_complete
    assert not report.real_market_data_rights_verified
    assert not report.real_forward_paper_allowed
    assert report.orders_sent==0
    with pytest.raises(ValueError,match="at least 205"):
        j.readiness(required_sessions=3)


def test_readiness_detects_symbol_universe_changes(tmp_path):
    j=PaperFeedJournal(tmp_path/"feed")
    j.stage(record("2025-01-02",symbols=("AAA","BBB")),
            clock_utc=observed("2025-01-02")+timedelta(hours=1))
    j.stage(record("2025-01-03",symbols=("AAA",)),
            clock_utc=observed("2025-01-03")+timedelta(hours=1))
    r=j.readiness()
    assert r.complete_xsto_calendar
    assert not r.constant_symbol_universe
    assert not r.technical_warmup_complete
    assert not r.real_forward_paper_allowed
