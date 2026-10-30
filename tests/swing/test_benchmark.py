"""Strict, synthetic benchmark comparison regression suite."""
from __future__ import annotations
from dataclasses import replace

import exchange_calendars as xcals
import pandas as pd
import pytest

from stockbot.swing.benchmark import IndexProvenance, compare_oos_to_index
from stockbot.swing.walkforward import WalkForwardFold, WalkForwardResult


def specimen():
    days = xcals.get_calendar("XSTO").sessions_in_range(
        "2025-01-02", "2025-06-30"
    ).tz_localize(None)
    assert len(days) > 45
    frame = pd.DataFrame({"date": days, "close": [100 + n*.1 for n in range(len(days))]})
    folds = []
    for start_i in (5, 25):
        folds.append(WalkForwardFold(
            number=len(folds)+1,
            train_end=str(days[start_i-1].date()),
            test_start=str(days[start_i].date()),
            test_end=str(days[start_i+19].date()),
            chosen_atr_stop_multiple=2.5,
            training_net_return_pct=2.0,
            test_net_return_pct=2.0,
            test_drawdown_pct=1.0,
            test_hypothetical_fills=2,
            test_commissions_sek=1.5,
            cost_stress_returns_pct={},
        ))
    result = WalkForwardResult(
        mode="WALK_FORWARD_BACKTEST_ONLY",
        data_quality="UNVERIFIED",
        data_sha256="a"*64,
        independent_benchmark_validated=False,
        money_traded_sek=0.0,
        fold_count=2,
        folds=tuple(folds),
        compounded_hypothetical_return_pct=4.04,
        stitched_max_drawdown_pct=1.0,
        total_hypothetical_fills=4,
        sufficient_activity_for_live_decision=False,
    )
    provenance = IndexProvenance(
        index_code="OMXSBGI", source="SYNTHETIC_TEST_DATA",
        usage_rights_reference="synthetic internal fixture",
        collected_at_utc="2026-10-29T22:00:00Z",
    )
    return frame, result, provenance


def test_benchmark_is_oos_aligned_total_return_only_and_not_alpha_proof():
    levels, result, provenance = specimen()
    answer = compare_oos_to_index(result, levels, provenance)
    assert answer.index_code == "OMXSBGI"
    assert answer.mode == "UNVERIFIED_OOS_BENCHMARK_COMPARISON"
    assert answer.license_independently_verified is False
    assert answer.external_index_history_verified is False
    assert answer.alpha_claim_permitted is False
    assert answer.chained_strategy_return_pct == pytest.approx(4.04)
    assert answer.folds[0].test_start == result.folds[0].test_start
    assert answer.folds[0].excess_return_pp == pytest.approx(
        2 - (levels.iloc[24]["close"] / levels.iloc[4]["close"] - 1)*100, abs=1e-6
    )


def test_rejects_missing_one_index_session_inside_fold():
    levels, result, source = specimen()
    missing = pd.Timestamp(result.folds[0].test_start)
    levels = levels[levels["date"] != missing]
    with pytest.raises(ValueError, match="missing XSTO index sessions"):
        compare_oos_to_index(result, levels, source)


def test_rejects_bad_dates_duplicate_prices_and_stale_timestamp():
    levels, result, source = specimen()
    with pytest.raises(ValueError, match="duplicate benchmark"):
        compare_oos_to_index(result, pd.concat([levels, levels.iloc[[0]]]), source)
    bad = levels.copy()
    bad.loc[12, "close"] = -1
    with pytest.raises(ValueError, match="invalid benchmark close"):
        compare_oos_to_index(result, bad, source)
    stale = replace(source, collected_at_utc="2025-01-03T08:00:00Z")
    with pytest.raises(ValueError, match="before last session"):
        compare_oos_to_index(result, levels, stale)


def test_no_price_only_index_misrepresented_as_dividend_inclusive():
    with pytest.raises(ValueError, match="gross total return"):
        IndexProvenance("OMXSBGI", "source", "ref", "2026-10-29T22:00:00Z",
                        return_basis="PRICE_ONLY")
    with pytest.raises(ValueError, match="supported Swedish"):
        IndexProvenance("OMXS30", "source", "ref", "2026-10-29T22:00:00Z")
    with pytest.raises(ValueError, match="UTC"):
        IndexProvenance("OMXSBGI", "source", "ref", "2026-10-29T22:00:00+02:00")


def test_benchmark_comparison_deterministic_and_independent_of_row_order():
    levels, result, source = specimen()
    expected = compare_oos_to_index(result, levels, source)
    actual = compare_oos_to_index(levels=levels.sample(frac=1, random_state=7),
                                  result=result, provenance=source)
    assert actual.benchmark_source_sha256 == expected.benchmark_source_sha256
    assert actual.folds == expected.folds
