from datetime import datetime, timezone
from types import SimpleNamespace

from stockbot.data.research_quality import ResearchDataAttestation, ResearchDataQualityReport
from stockbot.data.schemas import DataGrade
from stockbot.research.market_training import _snapshot_metadata


def _snapshot(grade: DataGrade):
    manifest = SimpleNamespace(
        provider="verified-provider",
        grade=grade,
        schema_version="1.2",
        created_at=datetime(2026, 9, 8, tzinfo=timezone.utc),
    )
    return SimpleNamespace(snapshot_id="snapshot-test", manifest=manifest)


def _passing_report():
    return ResearchDataQualityReport(
        canonical_schema_valid=True,
        adjusted_coverage=1.0,
        retrieval_causality_fraction=1.0,
        provider_count=1,
        providers=("verified-provider",),
        corporate_action_fields_present=True,
        universe_report=None,
        attestation=ResearchDataAttestation(
            adjusted_prices_verified=True,
            corporate_actions_complete=True,
            corporate_actions_point_in_time=True,
        ),
        research_grade_eligible=True,
        reasons=(),
    )


def test_declared_research_grade_without_quality_evidence_is_downgraded():
    metadata = _snapshot_metadata(_snapshot(DataGrade.RESEARCH_GRADE), None)
    assert metadata.grade is DataGrade.BOOTSTRAP


def test_declared_research_grade_with_verified_quality_remains_research_grade():
    metadata = _snapshot_metadata(_snapshot(DataGrade.RESEARCH_GRADE), _passing_report())
    assert metadata.grade is DataGrade.RESEARCH_GRADE


def test_bootstrap_snapshot_cannot_be_upgraded_by_quality_report():
    metadata = _snapshot_metadata(_snapshot(DataGrade.BOOTSTRAP), _passing_report())
    assert metadata.grade is DataGrade.BOOTSTRAP
