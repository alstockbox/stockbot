from datetime import datetime, timezone
import numpy as np
import pandas as pd

from stockbot.data.schemas import DataGrade
from stockbot.data.snapshots import SnapshotManifestInput, SnapshotStore
from stockbot.ml.models import ModelConfig
from stockbot.research.market_training import prepare_training_bars, train_snapshot


def _bars():
    rng = np.random.default_rng(99)
    dates = pd.date_range("2021-01-04", periods=260, freq="B", tz="UTC")
    rows = []
    for j, symbol in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]):
        ret = 0.0002 + 0.00005*j + rng.normal(0, 0.009, len(dates))
        raw = 100*np.exp(np.cumsum(ret))
        adj = raw * 0.5
        for i, dt in enumerate(dates):
            rows.append({
                "timestamp": dt, "symbol": symbol,
                "open": raw[i], "high": raw[i]*1.01, "low": raw[i]*0.99, "close": raw[i], "volume": 1_000_000+i,
                "adj_open": adj[i], "adj_high": adj[i]*1.01, "adj_low": adj[i]*0.99, "adj_close": adj[i], "adj_volume": 2_000_000+i,
                "div_cash": 0.0, "split_factor": 1.0, "provider": "fixture",
                "retrieved_at": pd.Timestamp("2026-01-01", tz="UTC"),
            })
    return pd.DataFrame(rows)


def test_prepare_training_bars_prefers_adjusted_market_fields():
    prepared = prepare_training_bars(_bars().iloc[:1])
    assert prepared.iloc[0]["close"] == _bars().iloc[0]["adj_close"]
    assert prepared.iloc[0]["volume"] == _bars().iloc[0]["adj_volume"]


def test_bootstrap_snapshot_runs_existing_alpha_arena_but_cannot_promote(tmp_path):
    bars = _bars()
    meta = SnapshotManifestInput(
        provider="fixture", grade=DataGrade.BOOTSTRAP,
        symbols=tuple(sorted(bars["symbol"].unique())), start="2021-01-04", end="2021-12-31",
        provenance={"kind": "bootstrap"},
    )
    snapshot = SnapshotStore(tmp_path).write(bars, meta, created_at=datetime(2026,1,1,tzinfo=timezone.utc))
    run = train_snapshot(snapshot, model_configs=[ModelConfig("ridge", seed=7)], horizon=5)
    assert len(run.leaderboard) == 1
    assert run.data_grade is DataGrade.BOOTSTRAP
    assert run.champion_candidate is None
