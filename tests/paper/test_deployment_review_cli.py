import json
from types import SimpleNamespace

import pytest

from stockbot.cli import paper_arena as paper_cli
from stockbot.data.schemas import DataGrade


def test_deployment_review_cli_assembles_verified_evidence(tmp_path, monkeypatch, capsys):
    artifact = SimpleNamespace(
        artifact_id="artifact-a",
        strategy_id="strategy-a",
        experiment_id="experiment-a",
        research_cycle_id="cycle-a",
        source_dataset_fingerprint="dataset-a",
        broker_execution_available=False,
    )
    research = SimpleNamespace(
        research_ready=True,
        data_grade=DataGrade.RESEARCH_GRADE,
        job_id="job-a",
        experiment_id="experiment-a",
        research_cycle_id="cycle-a",
        dataset_fingerprint="dataset-a",
        quarantine_start="2026-01-02",
        quality_fingerprint="quality-a",
    )
    audit = SimpleNamespace(
        audit_id="audit-a",
        strategy_id="strategy-a",
        experiment_id="experiment-a",
        quarantine_start="2026-01-02",
        passed=True,
    )
    paper = SimpleNamespace(
        strategy_id="strategy-a",
        model_artifact_id="artifact-a",
        research_cycle_id="cycle-a",
        sessions=72,
        live_span_days=63,
        live_eligible=True,
        frozen_provenance_complete=True,
        reasons=(),
    )
    provenance = SimpleNamespace(
        strategy_id="strategy-a",
        model_artifact_id="artifact-a",
        research_cycle_id="cycle-a",
        verified=True,
        artifact_verified=True,
        model_hash_verified=True,
        artifact_strategy_match=True,
        artifact_cycle_match=True,
        signal_snapshots_verified=72,
        realization_snapshots_verified=72,
        snapshot_timeline_verified=True,
        observations_verified=72,
        snapshots_verified=73,
        snapshot_fingerprints=("snapshot-a",),
        reasons=(),
        broker_execution_available=False,
    )
    review = SimpleNamespace(
        research_ready=True,
        research_grade=True,
        quarantine_audit_passed=True,
        paper_live_evidence_eligible=True,
        paper_frozen_provenance_complete=True,
        paper_external_provenance_verified=True,
        hard_risk_engine_required=True,
        broker_execution_available=False,
        eligible_for_manual_live_review=True,
        reasons=(),
    )

    monkeypatch.setattr(paper_cli, "load_frozen_shadow_artifact", lambda path: artifact)
    monkeypatch.setattr(
        paper_cli,
        "verify_research_evidence_bundle",
        lambda run_dir, *, artifact_manifest: research,
        raising=False,
    )

    def load_audit(path, *, quarantine_start, strategy_id, research_cycle_id):
        assert research_cycle_id == "cycle-a"
        return audit

    monkeypatch.setattr(
        paper_cli,
        "load_verified_quarantine_audit_record",
        load_audit,
        raising=False,
    )
    monkeypatch.setattr(
        paper_cli,
        "evaluate_paper_track",
        lambda records, *, criteria: paper,
    )
    monkeypatch.setattr(
        paper_cli,
        "verify_frozen_paper_provenance",
        lambda records, *, artifact_dir, snapshot_root: provenance,
    )
    monkeypatch.setattr(
        paper_cli,
        "evaluate_bound_deployment_review",
        lambda **kwargs: review,
        raising=False,
    )

    class _Ledger:
        def __init__(self, path):
            self.path = path

        def records(self, *, strategy_id):
            assert strategy_id == "strategy-a"
            return [SimpleNamespace(strategy_id=strategy_id)]

    monkeypatch.setattr(paper_cli, "PaperTradingLedger", _Ledger)

    report_path = tmp_path / "deployment-review.json"
    args = paper_cli.build_parser().parse_args(
        [
            "--ledger", str(tmp_path / "paper.jsonl"),
            "deployment-review",
            "--artifact", str(tmp_path / "artifact"),
            "--snapshot-root", str(tmp_path / "snapshots"),
            "--research-run", str(tmp_path / "run"),
            "--audit-ledger", str(tmp_path / "audit.json"),
            "--report", str(report_path),
            "--min-sessions", "60",
            "--min-span-days", "45",
        ]
    )

    assert paper_cli.run_from_args(args) == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["strategy_id"] == "strategy-a"
    assert payload["artifact_id"] == "artifact-a"
    assert payload["experiment_id"] == "experiment-a"
    assert payload["research_cycle_id"] == "cycle-a"
    assert payload["research_ready"] is True
    assert payload["research_grade"] is True
    assert payload["quarantine_audit_passed"] is True
    assert payload["paper_live_evidence_eligible"] is True
    assert payload["paper_external_provenance_verified"] is True
    assert payload["eligible_for_manual_live_review"] is True
    assert payload["hard_risk_engine_required"] is True
    assert payload["broker_execution_available"] is False
    assert payload["reasons"] == []

    output = capsys.readouterr().out
    assert "eligible_for_manual_live_review=yes" in output
    assert "broker_execution=disabled" in output


def test_deployment_review_cli_persists_fail_closed_error_report(tmp_path, monkeypatch):
    report_path = tmp_path / "deployment-review.json"

    def fail_review(args):
        raise ValueError("research cycle identity mismatch")

    monkeypatch.setattr(paper_cli, "run_from_args", fail_review)

    with pytest.raises(SystemExit) as exc_info:
        paper_cli.main(
            [
                "--ledger", str(tmp_path / "paper.jsonl"),
                "deployment-review",
                "--artifact", str(tmp_path / "artifact"),
                "--snapshot-root", str(tmp_path / "snapshots"),
                "--research-run", str(tmp_path / "run"),
                "--audit-ledger", str(tmp_path / "audit.json"),
                "--report", str(report_path),
            ]
        )

    assert exc_info.value.code == 2
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"
    assert payload["error"] == "research cycle identity mismatch"
    assert payload["eligible_for_manual_live_review"] is False
    assert payload["broker_execution_available"] is False
