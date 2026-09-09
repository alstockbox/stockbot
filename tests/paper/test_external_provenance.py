from types import SimpleNamespace

import pytest

from stockbot.paper.ledger import PaperObservation
from stockbot.paper.provenance import verify_frozen_paper_provenance


def _row(
    *,
    timestamp: str = "2026-09-02T16:00:00+00:00",
    signal_timestamp: str = "2026-09-01T16:00:00+00:00",
    artifact_id: str = "artifact-a",
    cycle_id: str = "cycle-a",
    signal_fingerprint: str = "snapshot-signal",
    realization_fingerprint: str = "snapshot-realization",
) -> PaperObservation:
    return PaperObservation(
        strategy_id="strategy-a",
        timestamp=timestamp,
        net_return=0.001,
        benchmark_return=0.0,
        turnover=0.2,
        cost_rate=0.0001,
        fill_rate=0.99,
        signal_count=2,
        research_cycle_id=cycle_id,
        model_artifact_id=artifact_id,
        signal_timestamp=signal_timestamp,
        signal_snapshot_fingerprint=signal_fingerprint,
        realization_snapshot_fingerprint=realization_fingerprint,
    )


def _snapshot(fingerprint: str, latest: str, *, symbols=("AAA", "BBB"), provenance=None):
    return SimpleNamespace(
        manifest=SimpleNamespace(
            dataset_fingerprint=fingerprint,
            symbols=tuple(symbols),
            last_observation={symbol: latest for symbol in symbols},
            provenance={} if provenance is None else dict(provenance),
        )
    )


def _install_verified_sources(monkeypatch, *, artifact_id="artifact-a", cycle_id="cycle-a", snapshots=None):
    import stockbot.paper.provenance as provenance_module

    manifest = SimpleNamespace(
        strategy_id="strategy-a",
        artifact_id=artifact_id,
        research_cycle_id=cycle_id,
        symbols=("AAA", "BBB"),
    )
    monkeypatch.setattr(
        provenance_module,
        "load_frozen_shadow_artifact",
        lambda artifact_dir: manifest,
    )

    snapshot_map = snapshots or {
        "snapshot-signal": _snapshot("snapshot-signal", "2026-09-01T16:00:00+00:00"),
        "snapshot-realization": _snapshot("snapshot-realization", "2026-09-02T16:00:00+00:00"),
    }

    class FakeSnapshotStore:
        def __init__(self, root):
            self.root = root

        def find_verified_by_fingerprint(self, fingerprint):
            return snapshot_map.get(fingerprint)

    monkeypatch.setattr(provenance_module, "SnapshotStore", FakeSnapshotStore)


def test_external_provenance_verifies_artifact_snapshots_and_timeline(monkeypatch, tmp_path):
    _install_verified_sources(monkeypatch)

    report = verify_frozen_paper_provenance(
        [_row()],
        artifact_dir=tmp_path / "artifact",
        snapshot_root=tmp_path / "snapshots",
    )

    assert report.verified
    assert report.strategy_id == "strategy-a"
    assert report.model_artifact_id == "artifact-a"
    assert report.research_cycle_id == "cycle-a"
    assert report.snapshots_verified == 2
    assert report.snapshot_fingerprints == ("snapshot-realization", "snapshot-signal")
    assert not report.broker_execution_available


def test_external_provenance_rejects_artifact_identity_mismatch(monkeypatch, tmp_path):
    _install_verified_sources(monkeypatch, artifact_id="artifact-other")

    with pytest.raises(ValueError, match="artifact"):
        verify_frozen_paper_provenance(
            [_row()],
            artifact_dir=tmp_path / "artifact",
            snapshot_root=tmp_path / "snapshots",
        )


def test_external_provenance_rejects_missing_snapshot_fingerprint(monkeypatch, tmp_path):
    _install_verified_sources(
        monkeypatch,
        snapshots={
            "snapshot-signal": _snapshot("snapshot-signal", "2026-09-01T16:00:00+00:00"),
        },
    )

    with pytest.raises(ValueError, match="snapshot fingerprint"):
        verify_frozen_paper_provenance(
            [_row()],
            artifact_dir=tmp_path / "artifact",
            snapshot_root=tmp_path / "snapshots",
        )


def test_external_provenance_rejects_snapshot_timeline_mismatch(monkeypatch, tmp_path):
    _install_verified_sources(
        monkeypatch,
        snapshots={
            "snapshot-signal": _snapshot("snapshot-signal", "2026-09-01T16:00:00+00:00"),
            "snapshot-realization": _snapshot("snapshot-realization", "2026-09-03T16:00:00+00:00"),
        },
    )

    with pytest.raises(ValueError, match="timestamp"):
        verify_frozen_paper_provenance(
            [_row()],
            artifact_dir=tmp_path / "artifact",
            snapshot_root=tmp_path / "snapshots",
        )


def test_external_provenance_rejects_snapshot_universe_mismatch(monkeypatch, tmp_path):
    _install_verified_sources(
        monkeypatch,
        snapshots={
            "snapshot-signal": _snapshot("snapshot-signal", "2026-09-01T16:00:00+00:00", symbols=("AAA",)),
            "snapshot-realization": _snapshot("snapshot-realization", "2026-09-02T16:00:00+00:00"),
        },
    )

    with pytest.raises(ValueError, match="universe"):
        verify_frozen_paper_provenance(
            [_row()],
            artifact_dir=tmp_path / "artifact",
            snapshot_root=tmp_path / "snapshots",
        )


def test_external_provenance_requires_continuous_forward_snapshot_chain(monkeypatch, tmp_path):
    rows = [
        _row(
            signal_fingerprint="snapshot-1",
            realization_fingerprint="snapshot-2",
        ),
        _row(
            timestamp="2026-09-03T16:00:00+00:00",
            signal_timestamp="2026-09-02T16:00:00+00:00",
            signal_fingerprint="snapshot-2-alternate",
            realization_fingerprint="snapshot-3",
        ),
    ]
    _install_verified_sources(
        monkeypatch,
        snapshots={
            "snapshot-1": _snapshot("snapshot-1", "2026-09-01T16:00:00+00:00"),
            "snapshot-2": _snapshot("snapshot-2", "2026-09-02T16:00:00+00:00"),
            "snapshot-2-alternate": _snapshot(
                "snapshot-2-alternate", "2026-09-02T16:00:00+00:00"
            ),
            "snapshot-3": _snapshot("snapshot-3", "2026-09-03T16:00:00+00:00"),
        },
    )

    with pytest.raises(ValueError, match="continuity"):
        verify_frozen_paper_provenance(
            rows,
            artifact_dir=tmp_path / "artifact",
            snapshot_root=tmp_path / "snapshots",
        )


def test_external_provenance_rejects_stale_symbol_inside_snapshot(monkeypatch, tmp_path):
    signal_snapshot = _snapshot(
        "snapshot-signal",
        "2026-09-01T16:00:00+00:00",
    )
    signal_snapshot.manifest.last_observation["BBB"] = "2026-08-31T16:00:00+00:00"
    _install_verified_sources(
        monkeypatch,
        snapshots={
            "snapshot-signal": signal_snapshot,
            "snapshot-realization": _snapshot(
                "snapshot-realization",
                "2026-09-02T16:00:00+00:00",
            ),
        },
    )

    with pytest.raises(ValueError, match="timestamp"):
        verify_frozen_paper_provenance(
            [_row()],
            artifact_dir=tmp_path / "artifact",
            snapshot_root=tmp_path / "snapshots",
        )


def test_external_provenance_rejects_signal_at_or_after_realization(monkeypatch, tmp_path):
    _install_verified_sources(
        monkeypatch,
        snapshots={
            "snapshot-signal": _snapshot("snapshot-signal", "2026-09-03T16:00:00+00:00"),
            "snapshot-realization": _snapshot("snapshot-realization", "2026-09-02T16:00:00+00:00"),
        },
    )

    with pytest.raises(ValueError, match="forward"):
        verify_frozen_paper_provenance(
            [_row(signal_timestamp="2026-09-03T16:00:00+00:00")],
            artifact_dir=tmp_path / "artifact",
            snapshot_root=tmp_path / "snapshots",
        )
