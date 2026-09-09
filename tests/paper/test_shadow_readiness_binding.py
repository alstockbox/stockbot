import pandas as pd

from stockbot.paper.runner import freeze_shadow_strategy, load_frozen_shadow_artifact
from stockbot.research.quarantine_audit import FrozenStrategySpec


def _bars() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=40, freq="B", tz="UTC")
    rows = []
    for offset, symbol in enumerate(("AAA", "BBB", "CCC")):
        for i, dt in enumerate(dates):
            close = 100.0 + 10.0 * offset + i * (0.2 + 0.01 * offset)
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": close * 0.999,
                    "high": close * 1.002,
                    "low": close * 0.998,
                    "close": close,
                    "volume": 1_000_000.0 + 10_000.0 * offset,
                }
            )
    return pd.DataFrame(rows)


def test_frozen_shadow_artifact_persists_research_readiness_fingerprint(tmp_path):
    spec = FrozenStrategySpec(
        strategy_id="strategy-readiness",
        experiment_id="experiment-readiness",
        horizon=1,
        model_name="ridge",
        model_params={"alpha": 1.0},
        seed=7,
        top_fraction=0.50,
        weighting="equal",
    )
    artifact_dir = tmp_path / "artifact"

    manifest = freeze_shadow_strategy(
        _bars(),
        spec,
        feature_names=("return_1", "momentum_5", "return_1_rank"),
        research_cycle_id="cycle-readiness",
        source_dataset_fingerprint="dataset-readiness",
        research_readiness_fingerprint="readiness-fingerprint-a",
        output_dir=artifact_dir,
    )

    assert manifest.research_readiness_fingerprint == "readiness-fingerprint-a"
    loaded = load_frozen_shadow_artifact(artifact_dir)
    assert loaded.research_readiness_fingerprint == "readiness-fingerprint-a"
    assert loaded.artifact_id == manifest.artifact_id
