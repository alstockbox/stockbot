import json
from types import SimpleNamespace

import pandas as pd

from stockbot.cli import paper_arena as paper_cli
from stockbot.data.schemas import DataGrade


def test_quarantine_audit_cli_uses_verified_research_lineage_and_snapshot(tmp_path, monkeypatch, capsys):
    artifact = SimpleNamespace(
        artifact_id="artifact-a",
        strategy_id="strategy-a",
        experiment_id="experiment-a",
        research_cycle_id="cycle-a",
        source_dataset_fingerprint="dataset-a",
        horizon=5,
        model_name="ridge",
        model_params={"alpha": 2.0},
        seed=11,
        top_fraction=0.25,
        weighting="equal",
        symbols=("AAA", "BBB"),
        broker_execution_available=False,
    )
    research = SimpleNamespace(
        research_ready=True,
        data_grade=DataGrade.RESEARCH_GRADE,
        job_id="job-a",
        experiment_id="experiment-a",
        research_cycle_id="cycle-a",
        dataset_fingerprint="dataset-a",
        quarantine_start="2026-01-05",
        quality_fingerprint="quality-a",
    )
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-05T00:00:00Z",
                    "2026-01-05T00:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
        }
    )
    snapshot = SimpleNamespace(
        bars=bars,
        manifest=SimpleNamespace(
            dataset_fingerprint="dataset-a",
            symbols=("AAA", "BBB"),
        ),
    )
    captured = {}

    monkeypatch.setattr(paper_cli, "load_frozen_shadow_artifact", lambda path: artifact)
    monkeypatch.setattr(
        paper_cli,
        "verify_research_evidence_bundle",
        lambda run_dir, *, artifact_manifest: research,
        raising=False,
    )

    class _SnapshotStore:
        def __init__(self, root):
            assert root == str(tmp_path / "snapshots")

        def find_verified_by_fingerprint(self, fingerprint):
            assert fingerprint == "dataset-a"
            return snapshot

    monkeypatch.setattr(paper_cli, "SnapshotStore", _SnapshotStore)

    def run_audit(audit_bars, quarantine_config, spec, *, ledger_path, holdout_config=None, research_cycle_id=None):
        captured["bars"] = audit_bars
        captured["quarantine_config"] = quarantine_config
        captured["spec"] = spec
        captured["ledger_path"] = ledger_path
        captured["research_cycle_id"] = research_cycle_id
        return SimpleNamespace(
            record=SimpleNamespace(
                audit_id="audit-a",
                research_cycle_id="cycle-a",
                passed=True,
                score=0.75,
                reasons=(),
            ),
            manifest=SimpleNamespace(
                quarantine_id="quarantine-a",
                start="2026-01-05T00:00:00+00:00",
            ),
        )

    monkeypatch.setattr(paper_cli, "run_single_quarantine_audit", run_audit, raising=False)

    report_path = tmp_path / "quarantine-audit.json"
    audit_ledger = tmp_path / "audit-ledger.json"
    args = paper_cli.build_parser().parse_args(
        [
            "quarantine-audit",
            "--artifact", str(tmp_path / "artifact"),
            "--snapshot-root", str(tmp_path / "snapshots"),
            "--research-run", str(tmp_path / "run"),
            "--audit-ledger", str(audit_ledger),
            "--report", str(report_path),
        ]
    )

    assert paper_cli.run_from_args(args) == 0
    assert captured["bars"] is snapshot.bars
    assert captured["quarantine_config"].start == "2026-01-05"
    assert captured["spec"].strategy_id == "strategy-a"
    assert captured["spec"].experiment_id == "experiment-a"
    assert captured["spec"].horizon == 5
    assert captured["spec"].model_name == "ridge"
    assert captured["spec"].model_params == {"alpha": 2.0}
    assert captured["spec"].seed == 11
    assert captured["spec"].top_fraction == 0.25
    assert captured["spec"].weighting == "equal"
    assert captured["ledger_path"] == str(audit_ledger)
    assert captured["research_cycle_id"] == "cycle-a"

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["audit_id"] == "audit-a"
    assert payload["strategy_id"] == "strategy-a"
    assert payload["experiment_id"] == "experiment-a"
    assert payload["research_cycle_id"] == "cycle-a"
    assert payload["dataset_fingerprint"] == "dataset-a"
    assert payload["quarantine_id"] == "quarantine-a"
    assert payload["passed"] is True
    assert payload["broker_execution_available"] is False

    output = capsys.readouterr().out
    assert "quarantine_audit_id=audit-a" in output
    assert "broker_execution=disabled" in output
