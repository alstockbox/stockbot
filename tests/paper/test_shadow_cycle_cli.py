import json
from types import SimpleNamespace

import pandas as pd
import pytest

import stockbot.cli.paper_arena as paper_cli


def _artifact():
    return SimpleNamespace(
        artifact_id="artifact-cycle",
        research_cycle_id="research-cycle-123",
        symbols=("AAA", "BBB"),
    )


def _snapshot(snapshot_id="snapshot-refreshed", fingerprint="fingerprint-refreshed"):
    return SimpleNamespace(
        snapshot_id=snapshot_id,
        bars=pd.DataFrame({"fixture": [1]}),
        manifest=SimpleNamespace(
            start="2025-01-01",
            dataset_fingerprint=fingerprint,
            symbols=("AAA", "BBB"),
        ),
    )


def _step_result(*, idempotent_replay=False, observation=None):
    return SimpleNamespace(
        strategy_id="strategy-shadow",
        artifact_id="artifact-cycle",
        processed_timestamp="2026-09-09T00:00:00+00:00",
        pending_signal_timestamp="2026-09-09T00:00:00+00:00",
        target_weights={"AAA": 0.5, "BBB": 0.5},
        observation=observation,
        idempotent_replay=idempotent_replay,
        broker_execution_available=False,
    )


def test_shadow_cycle_refreshes_exact_frozen_universe_and_steps_new_snapshot(tmp_path, monkeypatch, capsys):
    captured = {}
    artifact = _artifact()
    baseline = _snapshot("snapshot-baseline", "fingerprint-baseline")
    refreshed = _snapshot()
    provider = SimpleNamespace(name="fixture-provider")

    monkeypatch.setattr(paper_cli, "load_frozen_shadow_artifact", lambda path: artifact)
    monkeypatch.setattr(
        paper_cli,
        "_select_latest_compatible_snapshot",
        lambda root, symbols: baseline,
        raising=False,
    )
    monkeypatch.setattr(
        paper_cli,
        "resolve_shadow_provider",
        lambda name: provider,
        raising=False,
    )

    def fake_download(provider_arg, symbols, start, end, store, **kwargs):
        captured["provider"] = provider_arg
        captured["symbols"] = tuple(symbols)
        captured["start"] = str(start)
        captured["end"] = str(end)
        captured["store_root"] = store.root
        captured["download_kwargs"] = kwargs
        return refreshed

    def fake_step(bars, **kwargs):
        captured["step_bars"] = bars
        captured["step_kwargs"] = kwargs
        return _step_result()

    monkeypatch.setattr(paper_cli, "download_market_snapshot", fake_download, raising=False)
    monkeypatch.setattr(paper_cli, "run_shadow_step", fake_step)

    report_path = tmp_path / "paper" / "cycle-report.json"
    args = paper_cli.build_parser().parse_args(
        [
            "--ledger", str(tmp_path / "paper" / "observations.jsonl"),
            "cycle",
            "--artifact", str(tmp_path / "artifact"),
            "--snapshot-root", str(tmp_path / "snapshots"),
            "--provider", "yahoo-bootstrap",
            "--end", "2026-09-09",
            "--state", str(tmp_path / "paper" / "state.json"),
            "--report", str(report_path),
        ]
    )

    assert paper_cli.run_from_args(args) == 0
    assert captured["provider"] is provider
    assert captured["symbols"] == artifact.symbols
    assert captured["start"] == "2025-01-01"
    assert captured["end"] == "2026-09-09"
    assert captured["download_kwargs"]["provenance"]["shadow_artifact_id"] == artifact.artifact_id
    assert captured["download_kwargs"]["provenance"]["research_cycle_id"] == artifact.research_cycle_id
    assert captured["download_kwargs"]["reuse_identical"] is True
    assert captured["step_bars"] is refreshed.bars
    assert captured["step_kwargs"]["snapshot_fingerprint"] == refreshed.manifest.dataset_fingerprint

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == {
        "schema_version": 1,
        "status": "ok",
        "provider": "yahoo-bootstrap",
        "strategy_id": "strategy-shadow",
        "artifact_id": "artifact-cycle",
        "research_cycle_id": "research-cycle-123",
        "snapshot_id": "snapshot-refreshed",
        "snapshot_fingerprint": "fingerprint-refreshed",
        "snapshot_status": "new",
        "paper_step_status": "pending",
        "processed_timestamp": "2026-09-09T00:00:00+00:00",
        "pending_signal_timestamp": "2026-09-09T00:00:00+00:00",
        "paper_observation_timestamp": None,
        "target_count": 2,
        "broker_execution_available": False,
    }
    assert not report_path.with_suffix(".json.tmp").exists()

    output = capsys.readouterr().out
    assert "refreshed_snapshot_id=snapshot-refreshed" in output
    assert "refreshed_snapshot_fingerprint=fingerprint-refreshed" in output
    assert "broker_execution=disabled" in output


def test_shadow_cycle_requires_verified_baseline_when_start_is_not_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_cli, "load_frozen_shadow_artifact", lambda path: _artifact())
    monkeypatch.setattr(
        paper_cli,
        "_select_latest_compatible_snapshot",
        lambda root, symbols: (_ for _ in ()).throw(ValueError("no compatible verified snapshots found")),
        raising=False,
    )

    args = paper_cli.build_parser().parse_args(
        [
            "cycle",
            "--artifact", str(tmp_path / "artifact"),
            "--snapshot-root", str(tmp_path / "snapshots"),
            "--provider", "yahoo-bootstrap",
            "--end", "2026-09-09",
        ]
    )
    with pytest.raises(ValueError, match="compatible"):
        paper_cli.run_from_args(args)


def test_shadow_cycle_explicit_start_can_bootstrap_without_existing_snapshot(tmp_path, monkeypatch):
    captured = {}
    artifact = _artifact()
    refreshed = _snapshot()

    monkeypatch.setattr(paper_cli, "load_frozen_shadow_artifact", lambda path: artifact)
    monkeypatch.setattr(paper_cli, "resolve_shadow_provider", lambda name: SimpleNamespace(name="fixture"), raising=False)
    monkeypatch.setattr(
        paper_cli,
        "_select_latest_compatible_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("baseline lookup must not run with explicit start")),
        raising=False,
    )

    def fake_download(provider, symbols, start, end, store, **kwargs):
        captured["start"] = str(start)
        captured["end"] = str(end)
        captured["reuse_identical"] = kwargs.get("reuse_identical")
        return refreshed

    monkeypatch.setattr(paper_cli, "download_market_snapshot", fake_download, raising=False)
    monkeypatch.setattr(paper_cli, "run_shadow_step", lambda *args, **kwargs: _step_result())

    args = paper_cli.build_parser().parse_args(
        [
            "cycle",
            "--artifact", str(tmp_path / "artifact"),
            "--snapshot-root", str(tmp_path / "snapshots"),
            "--provider", "yahoo-bootstrap",
            "--start", "2025-01-01",
            "--end", "2026-09-09",
        ]
    )
    assert paper_cli.run_from_args(args) == 0
    assert captured == {"start": "2025-01-01", "end": "2026-09-09", "reuse_identical": True}


def test_shadow_cycle_report_distinguishes_reused_snapshot_and_idempotent_step(tmp_path, monkeypatch):
    artifact = _artifact()
    refreshed = _snapshot()
    snapshot_root = tmp_path / "snapshots"
    (snapshot_root / refreshed.snapshot_id).mkdir(parents=True)

    monkeypatch.setattr(paper_cli, "load_frozen_shadow_artifact", lambda path: artifact)
    monkeypatch.setattr(paper_cli, "resolve_shadow_provider", lambda name: SimpleNamespace(name="fixture"), raising=False)
    monkeypatch.setattr(paper_cli, "download_market_snapshot", lambda *args, **kwargs: refreshed, raising=False)
    monkeypatch.setattr(
        paper_cli,
        "run_shadow_step",
        lambda *args, **kwargs: _step_result(idempotent_replay=True),
    )

    report_path = tmp_path / "paper" / "cycle-report.json"
    args = paper_cli.build_parser().parse_args(
        [
            "cycle",
            "--artifact", str(tmp_path / "artifact"),
            "--snapshot-root", str(snapshot_root),
            "--provider", "yahoo-bootstrap",
            "--start", "2025-01-01",
            "--end", "2026-09-09",
            "--report", str(report_path),
        ]
    )
    assert paper_cli.run_from_args(args) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["snapshot_status"] == "reused"
    assert report["paper_step_status"] == "idempotent"
    assert report["paper_observation_timestamp"] is None
    assert report["status"] == "ok"


def test_shadow_cycle_report_marks_settled_observation(tmp_path, monkeypatch):
    artifact = _artifact()
    refreshed = _snapshot()
    observation = SimpleNamespace(timestamp="2026-09-09T00:00:00+00:00")

    monkeypatch.setattr(paper_cli, "load_frozen_shadow_artifact", lambda path: artifact)
    monkeypatch.setattr(paper_cli, "resolve_shadow_provider", lambda name: SimpleNamespace(name="fixture"), raising=False)
    monkeypatch.setattr(paper_cli, "download_market_snapshot", lambda *args, **kwargs: refreshed, raising=False)
    monkeypatch.setattr(
        paper_cli,
        "run_shadow_step",
        lambda *args, **kwargs: _step_result(observation=observation),
    )

    report_path = tmp_path / "paper" / "cycle-report.json"
    args = paper_cli.build_parser().parse_args(
        [
            "cycle",
            "--artifact", str(tmp_path / "artifact"),
            "--snapshot-root", str(tmp_path / "snapshots"),
            "--provider", "yahoo-bootstrap",
            "--start", "2025-01-01",
            "--end", "2026-09-09",
            "--report", str(report_path),
        ]
    )
    assert paper_cli.run_from_args(args) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["paper_step_status"] == "settled"
    assert report["paper_observation_timestamp"] == observation.timestamp


def test_shadow_cycle_main_writes_machine_readable_error_report(tmp_path, monkeypatch):
    report_path = tmp_path / "paper" / "cycle-report.json"
    monkeypatch.setattr(paper_cli, "load_frozen_shadow_artifact", lambda path: _artifact())
    monkeypatch.setattr(
        paper_cli,
        "resolve_shadow_provider",
        lambda name: (_ for _ in ()).throw(paper_cli.ProviderError("fixture provider failure")),
        raising=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        paper_cli.main(
            [
                "cycle",
                "--artifact", str(tmp_path / "artifact"),
                "--snapshot-root", str(tmp_path / "snapshots"),
                "--provider", "yahoo-bootstrap",
                "--start", "2025-01-01",
                "--end", "2026-09-09",
                "--report", str(report_path),
            ]
        )
    assert exc_info.value.code == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == {
        "schema_version": 1,
        "status": "error",
        "provider": "yahoo-bootstrap",
        "error_type": "ProviderError",
        "error": "fixture provider failure",
        "broker_execution_available": False,
    }
    assert not report_path.with_suffix(".json.tmp").exists()
