import hashlib
import json
from types import SimpleNamespace

import pytest

import stockbot.paper.deployment_gate as deployment_gate_module
from stockbot.data.schemas import DataGrade
from stockbot.paper.deployment_gate import evaluate_deployment_evidence
from stockbot.research.deep_feedback import build_research_cycle_id


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


def _write_research_bundle(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataset_fingerprint = "dataset-a"
    quarantine_start = "2026-01-02"
    universe_fingerprint = "universe-a"
    auxiliary_fingerprint = "auxiliary-a"
    quality_payload = {
        "canonical_schema_valid": True,
        "adjusted_coverage": 1.0,
        "retrieval_causality_fraction": 1.0,
        "provider_count": 1,
        "providers": ["fixture"],
        "corporate_action_fields_present": True,
        "universe_report": {
            "membership_coverage": 1.0,
            "point_in_time_membership": True,
            "survivorship_bias_controlled": True,
            "includes_delisted_securities": True,
            "reasons": [],
        },
        "attestation": {
            "adjusted_prices_verified": True,
            "corporate_actions_complete": True,
            "corporate_actions_point_in_time": True,
        },
        "research_grade_eligible": True,
        "reasons": [],
    }
    quality_raw = json.dumps(
        quality_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    quality_fingerprint = hashlib.sha256(quality_raw).hexdigest()
    research_cycle_id = build_research_cycle_id(
        dataset_fingerprint=dataset_fingerprint,
        quarantine_start=quarantine_start,
        universe_fingerprint=universe_fingerprint,
        auxiliary_fingerprint=auxiliary_fingerprint,
        quality_fingerprint=quality_fingerprint,
    )
    job_payload = {
        "job_id": "job-a",
        "snapshot_id": "snapshot-a",
        "dataset_fingerprint": dataset_fingerprint,
        "horizons": [5],
        "max_candidates": 10,
        "max_workers": 2,
        "memory_path": "research_memory/experiments.jsonl",
        "created_at": "2026-09-09T00:00:00+00:00",
        "quarantine_start": quarantine_start,
        "auxiliary_fingerprint": auxiliary_fingerprint,
        "auxiliary_features": ["policy_rate"],
        "universe_fingerprint": universe_fingerprint,
        "quality_fingerprint": quality_fingerprint,
        "schema_version": 4,
    }
    identity_payload = {
        key: job_payload[key]
        for key in (
            "snapshot_id",
            "dataset_fingerprint",
            "horizons",
            "max_candidates",
            "max_workers",
            "memory_path",
            "quarantine_start",
            "auxiliary_fingerprint",
            "auxiliary_features",
            "universe_fingerprint",
            "quality_fingerprint",
        )
    }
    identity_raw = json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    job_payload["job_id"] = hashlib.sha256(identity_raw).hexdigest()[:24]
    summary_payload = {
        "job_id": job_payload["job_id"],
        "champion_experiment_id": "experiment-a",
        "champion_paper_ready": True,
        "champion_paper_readiness_score": 0.91,
        "champion_paper_readiness_reasons": [],
        "schema_version": 3,
    }
    cycle_payload = {
        "schema_version": 1,
        "research_cycle_id": research_cycle_id,
        "dataset_fingerprint": dataset_fingerprint,
        "quarantine_start": quarantine_start,
        "universe_fingerprint": universe_fingerprint,
        "auxiliary_fingerprint": auxiliary_fingerprint,
        "quality_fingerprint": quality_fingerprint,
        "deep_feedback_path": "research_memory/deep-findings.jsonl",
        "deep_feedback_weight": 0.25,
    }
    persisted_quality = {
        **quality_payload,
        "quality_fingerprint": quality_fingerprint,
        "declared_grade": DataGrade.RESEARCH_GRADE.value,
        "effective_grade": DataGrade.RESEARCH_GRADE.value,
    }
    for name, payload in (
        ("job.json", job_payload),
        ("summary.json", summary_payload),
        ("research_cycle.json", cycle_payload),
        ("data_quality.json", persisted_quality),
    ):
        (run_dir / name).write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    artifact = SimpleNamespace(
        experiment_id="experiment-a",
        research_cycle_id=research_cycle_id,
        source_dataset_fingerprint=dataset_fingerprint,
    )
    return run_dir, artifact


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


def test_research_evidence_bundle_is_bound_to_frozen_artifact(tmp_path):
    run_dir, artifact = _write_research_bundle(tmp_path)

    evidence = deployment_gate_module.verify_research_evidence_bundle(
        run_dir,
        artifact_manifest=artifact,
    )
    assert evidence.research_ready is True
    assert evidence.data_grade is DataGrade.RESEARCH_GRADE
    assert evidence.experiment_id == artifact.experiment_id
    assert evidence.research_cycle_id == artifact.research_cycle_id
    assert evidence.quarantine_start == "2026-01-02"

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["champion_experiment_id"] = "different-experiment"
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="experiment"):
        deployment_gate_module.verify_research_evidence_bundle(
            run_dir,
            artifact_manifest=artifact,
        )


def test_deployment_gate_accepts_verified_persisted_quarantine_record():
    paper = SimpleNamespace(live_eligible=True, frozen_provenance_complete=True)
    provenance = _verified_provenance()
    audit_record = SimpleNamespace(passed=True)

    report = evaluate_deployment_evidence(
        research_ready=True,
        data_grade=DataGrade.RESEARCH_GRADE,
        quarantine_audit=None,
        quarantine_audit_record=audit_record,
        paper_report=paper,
        paper_provenance_report=provenance,
    )
    assert report.quarantine_audit_passed
    assert report.eligible_for_manual_live_review

    with pytest.raises(ValueError, match="one quarantine audit"):
        evaluate_deployment_evidence(
            research_ready=True,
            data_grade=DataGrade.RESEARCH_GRADE,
            quarantine_audit=SimpleNamespace(holdout_report=SimpleNamespace(passed=True)),
            quarantine_audit_record=audit_record,
            paper_report=paper,
            paper_provenance_report=provenance,
        )
