import json
from types import SimpleNamespace

import pandas as pd

import stockbot.cli.market_training as market_cli
from stockbot.data.schemas import DataGrade
from stockbot.data.universe import PointInTimeUniverse, UniverseManifest, UniverseMembership


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
                    "provider": "verified-test",
                    "retrieved_at": pd.Timestamp("2025-01-10T20:00:00Z"),
                }
            )
    return pd.DataFrame(rows)


def _universe() -> PointInTimeUniverse:
    return PointInTimeUniverse(
        (
            UniverseMembership("AAA", "2020-01-01"),
            UniverseMembership("BBB", "2020-01-01"),
            UniverseMembership("OLD", "2020-01-01", "2024-12-31", delisted=True),
        ),
        UniverseManifest(
            source="verified-universe",
            survivorship_bias_controlled=True,
            includes_delisted_securities=True,
            point_in_time_membership=True,
        ),
    )


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


def test_factory_runtime_binds_quality_evidence_to_job_training_and_artifact(tmp_path, monkeypatch):
    snapshot = SimpleNamespace(
        snapshot_id="snapshot-quality-test",
        path=tmp_path / "snapshot-quality-test",
        bars=_bars(),
        manifest=SimpleNamespace(
            provider="verified-test",
            grade=DataGrade.BOOTSTRAP,
            symbols=("AAA", "BBB"),
            row_count=6,
            dataset_fingerprint="dataset-fingerprint",
        ),
    )
    captured = {}

    monkeypatch.setattr(market_cli, "download_market_snapshot", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(market_cli, "load_point_in_time_universe", lambda path: _universe())

    def fake_factory(snapshot_arg, **kwargs):
        captured["quality_report"] = kwargs.get("quality_report")
        return _factory_report()

    monkeypatch.setattr(market_cli, "run_snapshot_factory", fake_factory)

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
            "--factory-run-dir", str(run_root),
            "--factory-universe-input", str(tmp_path / "universe.json"),
            "--factory-attest-adjusted-prices-verified",
            "--factory-attest-corporate-actions-complete",
            "--factory-attest-corporate-actions-point-in-time",
        ]
    )

    assert market_cli.run_from_args(args) == 0
    assert captured["quality_report"] is not None
    assert captured["quality_report"].research_grade_eligible

    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    job = json.loads((run_dirs[0] / "job.json").read_text(encoding="utf-8"))
    quality = json.loads((run_dirs[0] / "data_quality.json").read_text(encoding="utf-8"))

    assert len(job["quality_fingerprint"]) == 64
    assert quality["quality_fingerprint"] == job["quality_fingerprint"]
    assert quality["research_grade_eligible"] is True
    assert quality["declared_grade"] == DataGrade.BOOTSTRAP.value
    assert quality["effective_grade"] == DataGrade.BOOTSTRAP.value
