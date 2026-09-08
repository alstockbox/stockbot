import numpy as np
import pandas as pd

from stockbot.data.point_in_time_features import (
    AuxiliaryFeatureKind,
    PointInTimeFeatureManifest,
    PointInTimeFeatureObservation,
    PointInTimeFeatureStore,
)
from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.features.research_matrix import BASE_RESEARCH_FEATURE_COLUMNS
from stockbot.ml.models import ModelConfig
from stockbot.research.auxiliary_ablation import evaluate_auxiliary_feature_ablation


def _bars_and_store():
    rng = np.random.default_rng(20260908)
    dates = pd.date_range("2024-01-02", periods=220, freq="B", tz="UTC")
    rows = []
    market = rng.normal(0.0002, 0.004, len(dates))
    for j, symbol in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")):
        local = market + 0.0004 * np.sin(np.arange(len(dates)) / (8.0 + j)) + rng.normal(0.0, 0.003, len(dates))
        close = 100.0 * np.exp(np.cumsum(local))
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": float(close[i]),
                    "high": float(close[i] * 1.006),
                    "low": float(close[i] * 0.994),
                    "close": float(close[i]),
                    "volume": int(700_000 + 40_000 * j + 2_000 * (i % 17)),
                }
            )

    observations = []
    for i, dt in enumerate(dates[::20]):
        observations.extend(
            [
                PointInTimeFeatureObservation(
                    feature_name="policy_rate",
                    value=2.0 + 0.05 * i,
                    observation_time=dt.isoformat(),
                    available_time=dt.isoformat(),
                    source="macro",
                    kind=AuxiliaryFeatureKind.MACRO,
                    revision_id=f"rate-{i}",
                ),
                PointInTimeFeatureObservation(
                    feature_name="macro_cycle",
                    value=float(np.sin(i / 3.0)),
                    observation_time=dt.isoformat(),
                    available_time=dt.isoformat(),
                    source="macro",
                    kind=AuxiliaryFeatureKind.MACRO,
                    revision_id=f"cycle-{i}",
                ),
            ]
        )
    store = PointInTimeFeatureStore(
        tuple(observations),
        PointInTimeFeatureManifest(
            source="synthetic",
            point_in_time=True,
            revision_aware=True,
        ),
    )
    return pd.DataFrame(rows), store


def test_auxiliary_ablation_replays_full_and_removed_feature_contracts():
    bars, store = _bars_and_store()
    metadata = DatasetMetadata("aux-ablation", "synthetic", DataGrade.RESEARCH_GRADE)
    feature_columns = BASE_RESEARCH_FEATURE_COLUMNS + (
        "aux__policy_rate",
        "aux__macro_cycle",
    )
    report = evaluate_auxiliary_feature_ablation(
        bars,
        metadata,
        ModelConfig("ridge", {"alpha": 1.0}, seed=7),
        horizon=5,
        feature_columns=feature_columns,
        auxiliary_store=store,
        train_periods=100,
        test_periods=20,
        auxiliary_min_coverage=0.95,
    )

    assert report.auxiliary_features == ("policy_rate", "macro_cycle")
    assert len(report.results) == 2
    assert {row.feature_name for row in report.results} == {"policy_rate", "macro_cycle"}
    assert np.isfinite(report.baseline_score)
    assert np.isfinite(report.no_auxiliary_score)
    assert np.isfinite(report.aggregate_score_impact)
    assert all(np.isfinite(row.score_impact) for row in report.results)
