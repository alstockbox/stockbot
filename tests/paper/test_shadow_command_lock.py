import fcntl
from types import SimpleNamespace

import pandas as pd
import pytest

import stockbot.cli.paper_arena as paper_cli


def _artifact():
    return SimpleNamespace(
        artifact_id="artifact-lock",
        research_cycle_id="cycle-lock",
        symbols=("AAA", "BBB"),
    )


def _snapshot():
    return SimpleNamespace(
        snapshot_id="snapshot-lock",
        bars=pd.DataFrame({"fixture": [1]}),
        manifest=SimpleNamespace(
            start="2025-01-01",
            dataset_fingerprint="fingerprint-lock",
            symbols=("AAA", "BBB"),
        ),
    )


def _step_result():
    return SimpleNamespace(
        strategy_id="strategy-lock",
        artifact_id="artifact-lock",
        processed_timestamp="2026-09-09T00:00:00+00:00",
        pending_signal_timestamp="2026-09-09T00:00:00+00:00",
        target_weights={"AAA": 0.5, "BBB": 0.5},
        observation=None,
        idempotent_replay=False,
        broker_execution_available=False,
    )


def _cycle_args(tmp_path, lock_path):
    return paper_cli.build_parser().parse_args(
        [
            "cycle",
            "--artifact", str(tmp_path / "artifact"),
            "--snapshot-root", str(tmp_path / "snapshots"),
            "--provider", "yahoo-bootstrap",
            "--start", "2025-01-01",
            "--end", "2026-09-09",
            "--state", str(tmp_path / "paper" / "state.json"),
            "--lock", str(lock_path),
        ]
    )


def test_shadow_cycle_fails_closed_before_provider_when_lock_is_held(tmp_path, monkeypatch):
    lock_path = tmp_path / "paper" / "shadow.lock"
    lock_path.parent.mkdir(parents=True)
    holder = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    provider_calls = []

    monkeypatch.setattr(paper_cli, "load_frozen_shadow_artifact", lambda path: _artifact())
    monkeypatch.setattr(
        paper_cli,
        "resolve_shadow_provider",
        lambda name: provider_calls.append(name) or SimpleNamespace(name="fixture"),
        raising=False,
    )

    try:
        args = _cycle_args(tmp_path, lock_path)
        with pytest.raises(ValueError, match="already running"):
            paper_cli.run_from_args(args)
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()

    assert provider_calls == []
    assert not (tmp_path / "snapshots").exists()


def test_shadow_cycle_releases_lock_after_successful_command(tmp_path, monkeypatch):
    lock_path = tmp_path / "paper" / "shadow.lock"
    calls = []

    monkeypatch.setattr(paper_cli, "load_frozen_shadow_artifact", lambda path: _artifact())
    monkeypatch.setattr(paper_cli, "resolve_shadow_provider", lambda name: SimpleNamespace(name="fixture"), raising=False)
    monkeypatch.setattr(
        paper_cli,
        "download_market_snapshot",
        lambda *args, **kwargs: calls.append("download") or _snapshot(),
        raising=False,
    )
    monkeypatch.setattr(
        paper_cli,
        "run_shadow_step",
        lambda *args, **kwargs: calls.append("step") or _step_result(),
    )

    args = _cycle_args(tmp_path, lock_path)
    assert paper_cli.run_from_args(args) == 0
    assert paper_cli.run_from_args(args) == 0
    assert calls == ["download", "step", "download", "step"]

    probe = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
        probe.close()
