from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

import stockbot.cli.paper_arena as paper_cli


def _snapshot(*retrieved_at):
    return SimpleNamespace(
        bars=pd.DataFrame({"retrieved_at": list(retrieved_at)}),
        manifest=SimpleNamespace(dataset_fingerprint="snapshot-fingerprint"),
    )


def _args(**overrides):
    values = {
        "artifact": "artifact",
        "state": "paper/state.json",
        "ledger": "paper/observations.jsonl",
        "data_age_seconds": None,
        "kill_switch": False,
        "auxiliary_input": None,
        "auxiliary_features": None,
        "auxiliary_max_age_days": None,
        "auxiliary_min_coverage": 0.80,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_shadow_commands_default_data_age_to_verified_snapshot_freshness(tmp_path):
    args = paper_cli.build_parser().parse_args(
        [
            "auto-step",
            "--artifact", str(tmp_path / "artifact"),
            "--snapshot-root", str(tmp_path / "snapshots"),
        ]
    )
    assert args.data_age_seconds is None


def test_snapshot_data_age_uses_oldest_retrieval_timestamp():
    now = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
    snapshot = _snapshot(
        now - timedelta(seconds=125),
        now - timedelta(seconds=40),
    )
    assert paper_cli._snapshot_data_age_seconds(snapshot, now=now) == pytest.approx(125.0)


def test_snapshot_data_age_rejects_future_retrieval_timestamp():
    now = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
    snapshot = _snapshot(now + timedelta(seconds=1))
    with pytest.raises(ValueError, match="future"):
        paper_cli._snapshot_data_age_seconds(snapshot, now=now)


def test_execute_shadow_snapshot_uses_automatic_snapshot_age(monkeypatch):
    captured = {}
    now = datetime.now(timezone.utc)
    snapshot = _snapshot(now - timedelta(seconds=90))

    def fake_step(bars, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(paper_cli, "run_shadow_step", fake_step)
    paper_cli._execute_shadow_snapshot(_args(), snapshot)
    assert 85.0 <= captured["data_age_seconds"] <= 95.0


def test_execute_shadow_snapshot_explicit_age_override_wins(monkeypatch):
    captured = {}
    snapshot = _snapshot(datetime.now(timezone.utc) - timedelta(days=10))

    monkeypatch.setattr(
        paper_cli,
        "run_shadow_step",
        lambda bars, **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )
    paper_cli._execute_shadow_snapshot(_args(data_age_seconds=7.5), snapshot)
    assert captured["data_age_seconds"] == pytest.approx(7.5)
