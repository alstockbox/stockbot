from types import SimpleNamespace

import pytest

import stockbot.paper.deployment_gate as deployment_gate
from stockbot.data.schemas import DataGrade


def _research():
    return deployment_gate.VerifiedResearchEvidence(
        research_ready=True,
        data_grade=DataGrade.RESEARCH_GRADE,
        job_id="job-a",
        experiment_id="experiment-a",
        research_cycle_id="cycle-a",
        dataset_fingerprint="dataset-a",
        quarantine_start="2026-01-02",
        quality_fingerprint="quality-a",
    )


def _artifact():
    return SimpleNamespace(
        strategy_id="strategy-a",
        artifact_id="artifact-a",
        experiment_id="experiment-a",
        research_cycle_id="cycle-a",
        source_dataset_fingerprint="dataset-a",
        broker_execution_available=False,
    )


def _audit(**overrides):
    payload = {
        "strategy_id": "strategy-a",
        "experiment_id": "experiment-a",
        "research_cycle_id": "cycle-a",
        "quarantine_start": "2026-01-02",
        "passed": True,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _paper(**overrides):
    payload = {
        "strategy_id": "strategy-a",
        "model_artifact_id": "artifact-a",
        "research_cycle_id": "cycle-a",
        "live_eligible": True,
        "frozen_provenance_complete": True,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _provenance(**overrides):
    payload = {
        "strategy_id": "strategy-a",
        "model_artifact_id": "artifact-a",
        "research_cycle_id": "cycle-a",
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
        "broker_execution_available": False,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_bound_deployment_review_requires_one_lineage_across_all_evidence():
    report = deployment_gate.evaluate_bound_deployment_review(
        research_evidence=_research(),
        artifact_manifest=_artifact(),
        quarantine_audit_record=_audit(),
        paper_report=_paper(),
        paper_provenance_report=_provenance(),
    )

    assert report.eligible_for_manual_live_review
    assert report.hard_risk_engine_required
    assert report.broker_execution_available is False
    assert report.reasons == ()

    with pytest.raises(ValueError, match="audit experiment"):
        deployment_gate.evaluate_bound_deployment_review(
            research_evidence=_research(),
            artifact_manifest=_artifact(),
            quarantine_audit_record=_audit(experiment_id="different-experiment"),
            paper_report=_paper(),
            paper_provenance_report=_provenance(),
        )

    with pytest.raises(ValueError, match="audit research cycle"):
        deployment_gate.evaluate_bound_deployment_review(
            research_evidence=_research(),
            artifact_manifest=_artifact(),
            quarantine_audit_record=_audit(research_cycle_id="different-cycle"),
            paper_report=_paper(),
            paper_provenance_report=_provenance(),
        )

    with pytest.raises(ValueError, match="paper artifact"):
        deployment_gate.evaluate_bound_deployment_review(
            research_evidence=_research(),
            artifact_manifest=_artifact(),
            quarantine_audit_record=_audit(),
            paper_report=_paper(model_artifact_id="different-artifact"),
            paper_provenance_report=_provenance(),
        )
