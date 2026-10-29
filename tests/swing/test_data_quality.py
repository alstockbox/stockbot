from __future__ import annotations

import exchange_calendars as xcals
import pandas as pd
import pytest

from stockbot.swing.data_quality import XSTOProvenance, audit_xsto_data


def sample_xsto_bars() -> pd.DataFrame:
    sessions = xcals.get_calendar("XSTO").sessions_in_range(
        "2025-01-02", "2026-03-31"
    ).tz_localize(None)[:225]
    assert len(sessions) == 225
    return pd.DataFrame({
        "date": sessions,
        "symbol": ["VOLV.B"] * len(sessions),
        "open": [100.0 + i * 0.1 for i in range(len(sessions))],
        "high": [101.0 + i * 0.1 for i in range(len(sessions))],
        "low": [99.0 + i * 0.1 for i in range(len(sessions))],
        "close": [100.5 + i * 0.1 for i in range(len(sessions))],
        "volume": [150_000.0] * len(sessions),
    })


def sample_manifest(collected_at_utc="2026-10-29T12:00:00Z") -> XSTOProvenance:
    return XSTOProvenance(
        provider="Manual research input, synthetic tests",
        terms_reference="https://example.test/fake-data",
        collected_at_utc=collected_at_utc,
        currency="SEK",
        price_basis="RAW",
    )


def test_valid_xsto_structure_produces_unverified_receipt():
    bars = sample_xsto_bars()
    receipt = audit_xsto_data(bars, sample_manifest())
    assert receipt.exchange == "XSTO"
    assert receipt.bars == 225
    assert receipt.sessions == 225
    assert len(receipt.data_sha256) == 64
    assert receipt.mode == "BACKTEST_ONLY"
    assert receipt.order_execution_permitted is False
    assert receipt.research_grade == "UNVERIFIED_PROVENANCE_AND_ACTIONS"
    assert receipt.license_independently_verified is False
    assert receipt.corporate_actions_independently_verified is False


def test_rejects_closed_exchange_day():
    bars = sample_xsto_bars()
    bars.loc[0, "date"] = "2025-01-01"  # New Year's Day, market closed
    with pytest.raises(ValueError, match="non-XSTO session"):
        audit_xsto_data(bars, sample_manifest())


def test_rejects_missing_exchange_session():
    bars = sample_xsto_bars().drop(index=100)
    with pytest.raises(ValueError, match="omits XSTO session"):
        audit_xsto_data(bars, sample_manifest())


def test_rejects_future_data_unsupported_currency_and_missing_license_ref():
    bars = sample_xsto_bars()
    with pytest.raises(ValueError, match="precedes final"):
        audit_xsto_data(bars, sample_manifest("2025-02-01T12:00:00Z"))
    with pytest.raises(ValueError, match="SEK"):
        XSTOProvenance("provider", "terms", "2026-10-29T12:00:00Z", currency="USD")
    with pytest.raises(ValueError, match="provider and terms"):
        XSTOProvenance("provider", "", "2026-10-29T12:00:00Z")
    with pytest.raises(ValueError, match="UTC"):
        XSTOProvenance("provider", "terms", "2026-10-29T12:00:00+02:00")


def test_canonical_data_fingerprint_ignores_input_row_order():
    bars = sample_xsto_bars()
    receipt = audit_xsto_data(bars, sample_manifest())
    shuffled = bars.sample(frac=1.0, random_state=12)
    assert audit_xsto_data(shuffled, sample_manifest()).data_sha256 == receipt.data_sha256


def test_self_attested_corporate_actions_does_not_automatically_upgrade_grade():
    bars = sample_xsto_bars()
    manifest = XSTOProvenance(
        provider="Synthetic",
        terms_reference="https://example.test/fake-data",
        collected_at_utc="2026-10-29T12:00:00Z",
        corporate_actions_checked=True,
        historical_universe_checked=True,
    )
    receipt = audit_xsto_data(bars, manifest)
    assert receipt.research_grade == "UNVERIFIED_PROVENANCE_AND_ACTIONS"
    assert receipt.historical_universe_independently_verified is False
