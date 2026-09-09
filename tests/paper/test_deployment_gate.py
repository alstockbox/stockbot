from types import SimpleNamespace

from stockbot.data.schemas import DataGrade
from stockbot.paper.deployment_gate import evaluate_deployment_evidence


def _verified_provenance(**overrides):
    payload = {
        "verified": True,
        "artifact_verified": True,
        "model_hash_verified": True,
        "artifact_strategy_match": True,
        "artifact_cycle_match": True,
        "signal_snapshots_verified": 1,
        "realization_snapshots_verified": 1,
        "snapshot_timeline_verified": True,
        "observations_verified": 1,
        "reasons": (),
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_manual_live_review_requires_all_independent_evidence_layers():
    audit = SimpleNamespace(holdout_report=SimpleNamespace(passed=True))
    paper = SimpleNamespace(live_eligible=True, frozen_provenance_complete=True)
    provenance = _verified_provenance()
    report = evaluate_deployment_evidence(
        research_ready=True,
        data_grade=DataGrade.RESEARCH_GRADE,
        quarantine_audit=audit,
        paper_report=paper,
        paper_provenance_report=provenance,
    )

    assert report.eligible_for_manual_live_review
    assert report.paper_frozen_provenance_complete
    assert report.paper_external_provenance_verified
    assert report.hard_risk_engine_required
    assert not report.broker_execution_available
    assert report.reasons == ()


def test_missing_audit_and_forward_paper_blocks_manual_live_review():
    report = evaluate_deployment_evidence(
        research_ready=True,
        data_grade=DataGrade.BOOTSTRAP,
        quarantine_audit=None,
        paper_report=None,
    )

    assert not report.eligible_for_manual_live_review
    assert "data_not_research_grade" in report.reasons
    assert "sealed_quarantine_audit_not_passed" in report.reasons
    assert "forward_paper_evidence_not_ready" in report.reasons
    assert "forward_paper_provenance_not_verified" in report.reasons
    assert "forward_paper_external_provenance_not_verified" in report.reasons
    assert not report.broker_execution_available


def test_internal_lineage_flags_are_not_enough_without_external_verification():
    audit = SimpleNamespace(holdout_report=SimpleNamespace(passed=True))
    paper = SimpleNamespace(live_eligible=True, frozen_provenance_complete=True)

    report = evaluate_deployment_evidence(
        research_ready=True,
        data_grade=DataGrade.RESEARCH_GRADE,
        quarantine_audit=audit,
        paper_report=paper,
        paper_provenance_report=None,
    )

    assert report.paper_frozen_provenance_complete
    assert not report.paper_external_provenance_verified
    assert not report.eligible_for_manual_live_review
    assert "forward_paper_external_provenance_not_verified" in report.reasons


def test_aggregate_verified_flag_is_not_enough_when_granular_provenance_is_incomplete():
    audit = SimpleNamespace(holdout_report=SimpleNamespace(passed=True))
    paper = SimpleNamespace(live_eligible=True, frozen_provenance_complete=True)
    provenance = _verified_provenance(model_hash_verified=False)

    report = evaluate_deployment_evidence(
        research_ready=True,
        data_grade=DataGrade.RESEARCH_GRADE,
        quarantine_audit=audit,
        paper_report=paper,
        paper_provenance_report=provenance,
    )

    assert not report.paper_external_provenance_verified
    assert not report.eligible_for_manual_live_review
    assert "forward_paper_external_provenance_not_verified" in report.reasons
