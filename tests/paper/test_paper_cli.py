from types import SimpleNamespace

import stockbot.cli.paper_arena as paper_cli
from stockbot.cli.paper_arena import build_parser, run_from_args


def test_paper_cli_records_session_without_broker_execution(tmp_path, capsys):
    ledger = tmp_path / "paper.jsonl"
    parser = build_parser()
    args = parser.parse_args(
        [
            "--ledger",
            str(ledger),
            "record",
            "--strategy-id",
            "strategy-a",
            "--net-return",
            "0.001",
            "--timestamp",
            "2026-09-08T18:00:00+00:00",
        ]
    )
    assert run_from_args(args) == 0
    output = capsys.readouterr().out
    assert "paper_recorded=" in output
    assert "broker_execution=disabled" in output
    assert ledger.exists()


def test_paper_cli_steps_verified_snapshot_through_frozen_runner(tmp_path, monkeypatch, capsys):
    captured = {}
    snapshot = SimpleNamespace(
        bars="bars-fixture",
        manifest=SimpleNamespace(dataset_fingerprint="snapshot-fingerprint-002"),
    )

    class FakeSnapshotStore:
        def __init__(self, root):
            captured["snapshot_root"] = root

        def load(self, snapshot_id):
            captured["snapshot_id"] = snapshot_id
            return snapshot

    def fake_shadow_step(bars, **kwargs):
        captured["bars"] = bars
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            strategy_id="strategy-shadow",
            artifact_id="artifact-shadow",
            processed_timestamp="2026-09-09T00:00:00+00:00",
            pending_signal_timestamp="2026-09-09T00:00:00+00:00",
            target_weights={"AAA": 0.5, "BBB": 0.5},
            observation=SimpleNamespace(
                timestamp="2026-09-09T00:00:00+00:00",
                net_return=0.0012,
                fill_rate=0.98,
                cost_rate=0.0001,
            ),
            broker_execution_available=False,
        )

    monkeypatch.setattr(paper_cli, "SnapshotStore", FakeSnapshotStore)
    monkeypatch.setattr(paper_cli, "run_shadow_step", fake_shadow_step)

    ledger = tmp_path / "paper" / "observations.jsonl"
    state = tmp_path / "paper" / "state.json"
    parser = build_parser()
    args = parser.parse_args(
        [
            "--ledger", str(ledger),
            "step",
            "--artifact", str(tmp_path / "artifact"),
            "--snapshot-root", str(tmp_path / "snapshots"),
            "--snapshot-id", "snapshot-002",
            "--state", str(state),
            "--data-age-seconds", "15",
        ]
    )

    assert run_from_args(args) == 0
    assert captured["snapshot_id"] == "snapshot-002"
    assert captured["bars"] == "bars-fixture"
    assert captured["kwargs"]["snapshot_fingerprint"] == "snapshot-fingerprint-002"
    assert captured["kwargs"]["artifact_dir"] == str(tmp_path / "artifact")
    assert captured["kwargs"]["state_path"] == str(state)
    assert captured["kwargs"]["ledger_path"] == str(ledger)
    assert captured["kwargs"]["data_age_seconds"] == 15.0

    output = capsys.readouterr().out
    assert "shadow_strategy_id=strategy-shadow" in output
    assert "paper_net_return=0.001200" in output
    assert "broker_execution=disabled" in output
