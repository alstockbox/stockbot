from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import stockbot.cli.market_training as market_cli
from stockbot.data.schemas import DataGrade
from stockbot.research.deep_feedback import build_research_cycle_id


def _bars() -> pd.DataFrame:
    rows = []
    for symbol in ("AAA", "BBB"):
        for dt in pd.date_range("2025-01-02", periods=3, freq="B", tz="UTC"):
            rows.append(
                {
                    "timestamp": dt,
                    "symbol": symbol,
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.0,
                    "close": 101.0,
                    "volume": 1_000_000.0,
                    "adj_open": 100.0,
                    "adj_high": 102.0,
                    "adj_low": 99.0,
                    "adj_close": 101.0,
                    "adj_volume": 1_000_000.0,
                    "div_cash": 0.0,
                    "split_factor": 1.0,
                    "provider": "fixture",
                    "retrieved_at": pd.Timestamp("2025-01-10T20:00:00Z"),
                }
            )
    return pd.DataFrame(rows)


def _factory_report():
    return SimpleNamespace(
        experiments_run=0,
        candidates_passed=0,
        holdout_evaluated=0,
        promotion_occurred=False,
        champion_candidate=None,
        active_champion=None,
        ensemble_report=None,
        holdout_start=None,
        candidates=[],
    )


def _deep_report():
    return SimpleNamespace(candidate_count=0, candidates={}, stacking_reports={})


def test_factory_cli_binds_same_research_cycle_to_factory_and_deep_feedback(tmp_path, monkeypatch, capsys):
    snapshot = SimpleNamespace(
        snapshot_id="snapshot-deep-feedback",
        path=tmp_path / "snapshot-deep-feedback",
        bars=_bars(),
        manifest=SimpleNamespace(
            provider="fixture",
            grade=DataGrade.BOOTSTRAP,
            symbols=("AAA", "BBB"),
            row_count=6,
            dataset_fingerprint="dataset-cycle-fingerprint",
        ),
    )
    captured = {}

    monkeypatch.setattr(market_cli, "download_market_snapshot", lambda *args, **kwargs: snapshot)

    def fake_factory(snapshot_arg, **kwargs):
        captured["factory"] = kwargs
        return _factory_report()

    def fake_deep(snapshot_arg, report, **kwargs):
        captured["deep"] = kwargs
        return _deep_report()

    monkeypatch.setattr(market_cli, "run_snapshot_factory", fake_factory)
    monkeypatch.setattr(market_cli, "run_snapshot_deep_diagnostics", fake_deep)

    memory_path = tmp_path / "memory" / "experiments.jsonl"
    run_root = tmp_path / "runs"
    args = market_cli.build_parser().parse_args(
        [
            "--provider", "yahoo-bootstrap",
            "--symbols", "AAA,BBB",
            "--start", "2025-01-01",
            "--end", "2025-01-10",
            "--snapshot-root", str(tmp_path / "snapshots"),
            "--factory",
            "--factory-candidates", "1",
            "--factory-workers", "1",
            "--factory-memory", str(memory_path),
            "--factory-run-dir", str(run_root),
            "--factory-deep-diagnostics",
            "--factory-deep-feedback-weight", "0.30",
        ]
    )

    assert market_cli.run_from_args(args) == 0

    deep_path = memory_path.with_name("deep-findings.jsonl")
    factory_kwargs = captured["factory"]
    deep_kwargs = captured["deep"]
    assert Path(factory_kwargs["deep_feedback_path"]) == deep_path
    assert Path(deep_kwargs["deep_feedback_path"]) == deep_path
    assert factory_kwargs["research_cycle_id"] == deep_kwargs["research_cycle_id"]
    assert factory_kwargs["deep_feedback_weight"] == 0.30

    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    job = json.loads((run_dirs[0] / "job.json").read_text(encoding="utf-8"))
    expected_cycle = build_research_cycle_id(
        dataset_fingerprint="dataset-cycle-fingerprint",
        quarantine_start=None,
        universe_fingerprint=None,
        auxiliary_fingerprint=None,
        quality_fingerprint=job["quality_fingerprint"],
    )
    assert factory_kwargs["research_cycle_id"] == expected_cycle

    cycle = json.loads((run_dirs[0] / "research_cycle.json").read_text(encoding="utf-8"))
    assert cycle["schema_version"] == 1
    assert cycle["research_cycle_id"] == expected_cycle
    assert cycle["dataset_fingerprint"] == "dataset-cycle-fingerprint"
    assert cycle["quarantine_start"] is None
    assert cycle["universe_fingerprint"] is None
    assert cycle["auxiliary_fingerprint"] is None
    assert cycle["quality_fingerprint"] == job["quality_fingerprint"]
    assert cycle["deep_feedback_path"] == str(deep_path)
    assert cycle["deep_feedback_weight"] == 0.30
    assert "workers" not in cycle
    assert "max_candidates" not in cycle

    output = capsys.readouterr().out
    assert f"research_cycle_id={expected_cycle}" in output
    assert f"deep_feedback_path={deep_path}" in output


def test_factory_cli_rejects_invalid_deep_feedback_weight(tmp_path, monkeypatch):
    snapshot = SimpleNamespace(
        snapshot_id="snapshot-deep-feedback",
        path=tmp_path / "snapshot-deep-feedback",
        bars=_bars(),
        manifest=SimpleNamespace(
            provider="fixture",
            grade=DataGrade.BOOTSTRAP,
            symbols=("AAA", "BBB"),
            row_count=6,
            dataset_fingerprint="dataset-cycle-fingerprint",
        ),
    )
    monkeypatch.setattr(market_cli, "download_market_snapshot", lambda *args, **kwargs: snapshot)

    args = market_cli.build_parser().parse_args(
        [
            "--provider", "yahoo-bootstrap",
            "--symbols", "AAA,BBB",
            "--start", "2025-01-01",
            "--end", "2025-01-10",
            "--snapshot-root", str(tmp_path / "snapshots"),
            "--factory",
            "--factory-deep-feedback-weight", "1.1",
        ]
    )

    try:
        market_cli.run_from_args(args)
    except ValueError as exc:
        assert "deep feedback weight" in str(exc)
    else:
        raise AssertionError("invalid deep feedback weight should fail closed")
