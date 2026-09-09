from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import stockbot.cli.market_training as market_cli
from stockbot.data.schemas import DataGrade


def _snapshot(tmp_path):
    return SimpleNamespace(
        snapshot_id="snapshot-shadow-freeze",
        path=tmp_path / "snapshot-shadow-freeze",
        bars=pd.DataFrame({"fixture": [1]}),
        manifest=SimpleNamespace(
            provider="fixture",
            grade=DataGrade.BOOTSTRAP,
            symbols=("AAA", "BBB"),
            row_count=2,
            dataset_fingerprint="dataset-shadow-freeze",
        ),
    )


def _candidate():
    artifact = SimpleNamespace(
        feature_names=("return_1", "momentum_5", "return_1_rank"),
    )
    result = SimpleNamespace(artifact=artifact)
    return SimpleNamespace(
        experiment_id="exp-shadow-freeze",
        horizon=5,
        model_name="ridge",
        model_params={"alpha": 2.0},
        seed=11,
        factory_score=0.72,
        selection_score=0.71,
        promotion_score=0.70,
        oos_coverage=0.90,
        stress_score=0.80,
        gate=SimpleNamespace(passed=True),
        holdout_report=SimpleNamespace(passed=True),
        discovery=None,
        regime_report=None,
        drift_report=None,
        paper_readiness=SimpleNamespace(ready=True),
        result=result,
    )


def _report(candidate):
    return SimpleNamespace(
        experiments_run=1,
        candidates_passed=1,
        holdout_evaluated=1,
        promotion_occurred=True,
        champion_candidate=candidate,
        active_champion=None,
        ensemble_report=None,
        holdout_start=None,
        candidates=(candidate,),
    )


def test_factory_cli_auto_freezes_exact_champion_contract(tmp_path, monkeypatch, capsys):
    snapshot = _snapshot(tmp_path)
    candidate = _candidate()
    captured = {}

    monkeypatch.setattr(market_cli, "download_market_snapshot", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(market_cli, "run_snapshot_factory", lambda *args, **kwargs: _report(candidate))
    monkeypatch.setattr(
        market_cli,
        "_evaluate_factory_data_quality",
        lambda *args, **kwargs: (
            SimpleNamespace(research_grade_eligible=False, reasons=("fixture",)),
            "quality-shadow-freeze",
            DataGrade.BOOTSTRAP,
        ),
    )

    def fake_freeze(bars, spec, **kwargs):
        captured["bars"] = bars
        captured["spec"] = spec
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            artifact_id="artifact-shadow-freeze",
            strategy_id=spec.strategy_id,
            broker_execution_available=False,
        )

    monkeypatch.setattr(market_cli, "freeze_shadow_strategy", fake_freeze, raising=False)

    artifact_dir = tmp_path / "paper" / "artifact"
    args = market_cli.build_parser().parse_args(
        [
            "--provider", "yahoo-bootstrap",
            "--symbols", "AAA,BBB",
            "--start", "2025-01-01",
            "--end", "2026-09-09",
            "--snapshot-root", str(tmp_path / "snapshots"),
            "--factory",
            "--factory-candidates", "1",
            "--factory-workers", "1",
            "--factory-freeze-shadow-dir", str(artifact_dir),
        ]
    )

    assert market_cli.run_from_args(args) == 0
    assert captured["bars"] is snapshot.bars
    spec = captured["spec"]
    assert spec.experiment_id == "exp-shadow-freeze"
    assert spec.horizon == 5
    assert spec.model_name == "ridge"
    assert spec.model_params == {"alpha": 2.0}
    assert spec.seed == 11
    assert spec.top_fraction == 0.30
    assert spec.weighting == "equal"
    assert captured["kwargs"]["feature_names"] == candidate.result.artifact.feature_names
    assert captured["kwargs"]["source_dataset_fingerprint"] == "dataset-shadow-freeze"
    assert captured["kwargs"]["output_dir"] == artifact_dir
    assert captured["kwargs"]["research_cycle_id"]
    assert captured["kwargs"]["execution_config"].capital == 100_000.0

    output = capsys.readouterr().out
    assert "shadow_artifact_id=artifact-shadow-freeze" in output
    assert "shadow_broker_execution=disabled" in output


def test_factory_cli_does_not_freeze_when_no_champion_exists(tmp_path, monkeypatch, capsys):
    snapshot = _snapshot(tmp_path)
    monkeypatch.setattr(market_cli, "download_market_snapshot", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(market_cli, "run_snapshot_factory", lambda *args, **kwargs: _report(None))
    monkeypatch.setattr(
        market_cli,
        "_evaluate_factory_data_quality",
        lambda *args, **kwargs: (
            SimpleNamespace(research_grade_eligible=False, reasons=("fixture",)),
            "quality-shadow-freeze",
            DataGrade.BOOTSTRAP,
        ),
    )

    called = []
    monkeypatch.setattr(market_cli, "freeze_shadow_strategy", lambda *args, **kwargs: called.append(True), raising=False)

    args = market_cli.build_parser().parse_args(
        [
            "--provider", "yahoo-bootstrap",
            "--symbols", "AAA,BBB",
            "--start", "2025-01-01",
            "--end", "2026-09-09",
            "--snapshot-root", str(tmp_path / "snapshots"),
            "--factory",
            "--factory-candidates", "1",
            "--factory-workers", "1",
            "--factory-freeze-shadow-dir", str(tmp_path / "artifact"),
        ]
    )

    assert market_cli.run_from_args(args) == 0
    assert called == []
    output = capsys.readouterr().out
    assert "shadow_freeze=skipped_no_champion" in output
