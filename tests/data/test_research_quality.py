import pandas as pd

from stockbot.data.research_quality import (
    ResearchDataAttestation,
    evaluate_research_data_quality,
    research_data_quality_fingerprint,
    verified_data_grade,
)
from stockbot.data.schemas import DataGrade
from stockbot.data.universe import PointInTimeUniverse, UniverseManifest, UniverseMembership


def _bars(*, causal_retrieval: bool = True) -> pd.DataFrame:
    rows = []
    retrieved = "2025-01-10T20:00:00Z" if causal_retrieval else "2024-01-01T00:00:00Z"
    for symbol in ("AAA", "BBB"):
        for dt in pd.date_range("2025-01-02", periods=5, freq="B", tz="UTC"):
            rows.append(
                {
                    "timestamp": dt,
                    "symbol": symbol,
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.0,
                    "close": 101.0,
                    "volume": 1_000_000.0,
                    "adj_open": 100.0,
                    "adj_high": 102.0,
                    "adj_low": 99.0,
                    "adj_close": 101.0,
                    "adj_volume": 1_000_000.0,
                    "div_cash": 0.0,
                    "split_factor": 1.0,
                    "provider": "verified-test",
                    "retrieved_at": retrieved,
                }
            )
    return pd.DataFrame(rows)


def _universe() -> PointInTimeUniverse:
    return PointInTimeUniverse(
        [
            UniverseMembership("AAA", "2020-01-01"),
            UniverseMembership("BBB", "2020-01-01"),
            UniverseMembership("OLD", "2020-01-01", "2024-12-31", delisted=True),
        ],
        UniverseManifest(
            source="verified-universe",
            survivorship_bias_controlled=True,
            includes_delisted_securities=True,
            point_in_time_membership=True,
        ),
    )


def _attestation() -> ResearchDataAttestation:
    return ResearchDataAttestation(
        adjusted_prices_verified=True,
        corporate_actions_complete=True,
        corporate_actions_point_in_time=True,
    )


def test_complete_attested_point_in_time_dataset_is_research_grade_eligible():
    report = evaluate_research_data_quality(
        _bars(),
        universe=_universe(),
        attestation=_attestation(),
    )
    assert report.research_grade_eligible
    assert report.adjusted_coverage == 1.0
    assert report.retrieval_causality_fraction == 1.0
    assert report.universe_report is not None
    assert report.universe_report.research_grade_universe
    assert verified_data_grade(DataGrade.RESEARCH_GRADE, report) is DataGrade.RESEARCH_GRADE


def test_quality_fingerprint_is_deterministic_and_changes_with_evidence():
    eligible = evaluate_research_data_quality(
        _bars(),
        universe=_universe(),
        attestation=_attestation(),
    )
    same = evaluate_research_data_quality(
        _bars(),
        universe=_universe(),
        attestation=_attestation(),
    )
    missing_attestation = evaluate_research_data_quality(
        _bars(),
        universe=_universe(),
    )
    assert research_data_quality_fingerprint(eligible) == research_data_quality_fingerprint(same)
    assert research_data_quality_fingerprint(eligible) != research_data_quality_fingerprint(missing_attestation)


def test_missing_attestation_and_universe_fail_closed():
    report = evaluate_research_data_quality(_bars())
    assert not report.research_grade_eligible
    assert "adjusted_prices_not_verified" in report.reasons
    assert "corporate_actions_not_attested_complete" in report.reasons
    assert "point_in_time_universe_missing" in report.reasons
    assert verified_data_grade(DataGrade.RESEARCH_GRADE, report) is DataGrade.BOOTSTRAP


def test_noncausal_retrieval_timestamp_blocks_research_grade():
    report = evaluate_research_data_quality(
        _bars(causal_retrieval=False),
        universe=_universe(),
        attestation=_attestation(),
    )
    assert not report.research_grade_eligible
    assert "non_causal_retrieval_timestamps" in report.reasons


def test_non_research_declared_grade_is_never_upgraded_by_quality_report():
    report = evaluate_research_data_quality(
        _bars(),
        universe=_universe(),
        attestation=_attestation(),
    )
    assert verified_data_grade(DataGrade.BOOTSTRAP, report) is DataGrade.BOOTSTRAP
