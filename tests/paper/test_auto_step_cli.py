from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

import stockbot.cli.paper_arena as paper_cli
from stockbot.data.schemas import DataGrade
from stockbot.data.snapshots import SnapshotManifestInput, SnapshotStore


def _bars(symbols, dates):
    rows = []
    retrieved_at = pd.Timestamp("2026-03-01", tz="UTC")
    for offset, symbol in enumerate(symbols):
        for i, dt in enumerate(pd.DatetimeIndex(dates)):
            px = 100.0 + 10.0 * offset + i
            rows.append(
                {
                    "timestamp": dt,
                    "symbol": symbol,
                    "open": px,
                    "high": px * 1.01,
                    "low": px * 0.99,
                    "close": px,
                    "volume": 1_000_000.0,
                    "adj_open": px,
                    "adj_high": px * 1.01,
                    "adj_low": px * 0.99,
                    "adj_close": px,
                    "adj_volume": 1_000_000.0,
                    "div_cash": 0.0,
                    "split_factor": 1.0,
                    "provider": "fixture",
                    "retrieved_at": retrieved_at,
                }
            )
    return pd.DataFrame(rows)


def _write_snapshot(store, *, symbols, dates, created_at):
    bars = _bars(symbols, dates)
    manifest = SnapshotManifestInput(
        provider="fixture",
        grade=DataGrade.BOOTSTRAP,
        symbols=tuple(symbols),
        start=pd.Timestamp(min(dates)).date().isoformat(),
        end=pd.Timestamp(max(dates)).date().isoformat(),
        provenance={"fixture": "auto-step"},
    )
    return store.write(bars, manifest, created_at=created_at)


def _result():
    return SimpleNamespace(
        strategy_id="strategy-shadow",
        artifact_id="artifact-shadow",
        processed_timestamp="2026-01-08T00:00:00+00:00",
        pending_signal_timestamp="2026-01-08T00:00:00+00:00",
        target_weights={"AAA": 0.5, "BBB": 0.5},
        observation=None,
        idempotent_replay=False,
        broker_execution_available=False,
    )


def test_auto_step_selects_newest_verified_compatible_snapshot(tmp_path, monkeypatch, capsys):
    root = tmp_path / "snapshots"
    store = SnapshotStore(root)
    older = _write_snapshot(
        store,
        symbols=("AAA", "BBB"),
        dates=pd.date_range("2026-01-02", periods=2, freq="B", tz="UTC"),
        created_at=datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc),
    )
    _write_snapshot(
        store,
        symbols=("ZZZ",),
        dates=pd.date_range("2026-01-09", periods=2, freq="B", tz="UTC"),
        created_at=datetime(2026, 1, 12, 18, 0, tzinfo=timezone.utc),
    )
    newest = _write_snapshot(
        store,
        symbols=("AAA", "BBB"),
        dates=pd.date_range("2026-01-07", periods=2, freq="B", tz="UTC"),
        created_at=datetime(2026, 1, 8, 18, 0, tzinfo=timezone.utc),
    )
    assert newest.snapshot_id != older.snapshot_id

    captured = {}
    monkeypatch.setattr(
        paper_cli,
        "load_frozen_shadow_artifact",
        lambda path: SimpleNamespace(symbols=("AAA", "BBB")),
        raising=False,
    )

    def fake_step(bars, **kwargs):
        captured["bars"] = bars
        captured["kwargs"] = kwargs
        return _result()

    monkeypatch.setattr(paper_cli, "run_shadow_step", fake_step)

    args = paper_cli.build_parser().parse_args(
        [
            "--ledger", str(tmp_path / "paper" / "observations.jsonl"),
            "auto-step",
            "--artifact", str(tmp_path / "artifact"),
            "--snapshot-root", str(root),
            "--state", str(tmp_path / "paper" / "state.json"),
        ]
    )
    assert paper_cli.run_from_args(args) == 0
    assert captured["kwargs"]["snapshot_fingerprint"] == newest.manifest.dataset_fingerprint
    assert pd.to_datetime(captured["bars"]["timestamp"], utc=True).max() == pd.Timestamp("2026-01-08", tz="UTC")

    output = capsys.readouterr().out
    assert f"selected_snapshot_id={newest.snapshot_id}" in output
    assert f"selected_snapshot_fingerprint={newest.manifest.dataset_fingerprint}" in output
    assert "broker_execution=disabled" in output


def test_auto_step_fails_closed_when_no_compatible_snapshot_exists(tmp_path, monkeypatch):
    root = tmp_path / "snapshots"
    store = SnapshotStore(root)
    _write_snapshot(
        store,
        symbols=("ZZZ",),
        dates=pd.date_range("2026-01-02", periods=2, freq="B", tz="UTC"),
        created_at=datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        paper_cli,
        "load_frozen_shadow_artifact",
        lambda path: SimpleNamespace(symbols=("AAA", "BBB")),
        raising=False,
    )

    args = paper_cli.build_parser().parse_args(
        [
            "auto-step",
            "--artifact", str(tmp_path / "artifact"),
            "--snapshot-root", str(root),
        ]
    )
    with pytest.raises(ValueError, match="compatible"):
        paper_cli.run_from_args(args)


def test_auto_step_does_not_silently_fall_back_from_corrupt_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "snapshots"
    store = SnapshotStore(root)
    _write_snapshot(
        store,
        symbols=("AAA", "BBB"),
        dates=pd.date_range("2026-01-02", periods=2, freq="B", tz="UTC"),
        created_at=datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc),
    )
    newest = _write_snapshot(
        store,
        symbols=("AAA", "BBB"),
        dates=pd.date_range("2026-01-07", periods=2, freq="B", tz="UTC"),
        created_at=datetime(2026, 1, 8, 18, 0, tzinfo=timezone.utc),
    )
    bars_path = newest.path / "bars.csv"
    bars_path.write_text(bars_path.read_text(encoding="utf-8").replace("101.0", "999.0", 1), encoding="utf-8")

    monkeypatch.setattr(
        paper_cli,
        "load_frozen_shadow_artifact",
        lambda path: SimpleNamespace(symbols=("AAA", "BBB")),
        raising=False,
    )

    args = paper_cli.build_parser().parse_args(
        [
            "auto-step",
            "--artifact", str(tmp_path / "artifact"),
            "--snapshot-root", str(root),
        ]
    )
    with pytest.raises(ValueError, match="fingerprint"):
        paper_cli.run_from_args(args)
