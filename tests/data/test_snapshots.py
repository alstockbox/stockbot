from datetime import datetime, timezone
import json

import pandas as pd
import pytest

from stockbot.data.schemas import DataGrade
from stockbot.data.snapshots import SnapshotManifestInput, SnapshotStore, compute_snapshot_fingerprint


def _bars(symbols=("AAA", "BBB")):
    rows = []
    retrieved = pd.Timestamp("2026-01-10", tz="UTC")
    for symbol in symbols:
        for dt, px in zip(pd.date_range("2026-01-02", periods=2, freq="B", tz="UTC"), (10.0, 11.0)):
            rows.append({
                "timestamp": dt, "symbol": symbol,
                "open": px, "high": px * 1.01, "low": px * 0.99, "close": px, "volume": 1000,
                "adj_open": px/2, "adj_high": px*1.01/2, "adj_low": px*0.99/2, "adj_close": px/2,
                "adj_volume": 2000, "div_cash": 0.0, "split_factor": 1.0,
                "provider": "fixture", "retrieved_at": retrieved,
            })
    return pd.DataFrame(rows)


def _input():
    return SnapshotManifestInput(
        provider="fixture",
        grade=DataGrade.BOOTSTRAP,
        symbols=("BBB", "AAA"),
        start="2026-01-01",
        end="2026-01-31",
        provenance={"universe": "test"},
    )


def test_snapshot_fingerprint_is_deterministic_and_ignores_retrieval_time():
    bars = _bars()
    first = compute_snapshot_fingerprint(bars, _input())
    changed = bars.copy()
    changed["retrieved_at"] = pd.Timestamp("2026-02-01", tz="UTC")
    second = compute_snapshot_fingerprint(changed, _input())
    assert first == second


def test_snapshot_store_round_trips_manifest_and_bars_without_secrets(tmp_path):
    store = SnapshotStore(tmp_path)
    created = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
    snapshot = store.write(_bars(), _input(), created_at=created)
    loaded = store.load(snapshot.snapshot_id)
    assert loaded.manifest.symbols == ("AAA", "BBB")
    assert loaded.manifest.row_count == 4
    assert loaded.manifest.grade is DataGrade.BOOTSTRAP
    assert loaded.manifest.dataset_fingerprint == snapshot.manifest.dataset_fingerprint
    manifest_text = (snapshot.path / "manifest.json").read_text()
    assert "token" not in manifest_text.lower() and "authorization" not in manifest_text.lower()
    assert set(json.loads(manifest_text)) >= {"snapshot_id", "provider", "grade", "dataset_fingerprint"}
    assert len(loaded.bars) == 4


def test_snapshot_store_never_overwrites_existing_snapshot_directory(tmp_path):
    store = SnapshotStore(tmp_path)
    created = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
    store.write(_bars(), _input(), created_at=created)
    with pytest.raises(FileExistsError):
        store.write(_bars(), _input(), created_at=created)


def test_snapshot_round_trip_is_stable_with_missing_adjusted_fields(tmp_path):
    bars = _bars(("AAA",))
    for column in ("adj_open", "adj_high", "adj_low", "adj_volume"):
        bars[column] = None
    manifest = SnapshotManifestInput(
        provider="yahoo-bootstrap", grade=DataGrade.BOOTSTRAP, symbols=("AAA",),
        start="2026-01-01", end="2026-01-31", provenance={"source": "bootstrap"},
    )
    snapshot = SnapshotStore(tmp_path).write(bars, manifest, created_at=datetime(2026,1,10,12,0,tzinfo=timezone.utc))
    loaded = SnapshotStore(tmp_path).load(snapshot.snapshot_id)
    assert loaded.manifest.dataset_fingerprint == snapshot.manifest.dataset_fingerprint
