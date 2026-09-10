import numpy as np
import pandas as pd

from stockbot.data.point_in_time_features import (
    AuxiliaryFeatureKind,
    PointInTimeFeatureManifest,
    PointInTimeFeatureObservation,
    PointInTimeFeatureStore,
)
from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.ml.models import ModelConfig
from stockbot.research.training_pipeline import run_training_research


def _bars():
    rng = np.random.default_rng(20260908)
    dates = pd.date_range("2024-01-02", periods=220, freq="B", tz="UTC")
    rows = []
    market = rng.normal(0.0002, 0.004, len(dates))
    for j, symbol in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")):
        local = market + 0.0005 * np.sin(np.arange(len(dates)) / (8.0 + j)) + rng.normal(0.0, 0.003, len(dates))
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
                    "volume": int(700_000 + 50_000 * j + 2_000 * (i % 17)),
                }
            )
    return pd.DataFrame(rows), dates


def _store(dates):
    observations = []
    for i, dt in enumerate(dates[::20]):
        observations.append(
            PointInTimeFeatureObservation(
                feature_name="policy_rate",
                value=2.0 + 0.05 * i,
                observation_time=dt.isoformat(),
                available_time=dt.isoformat(),
                source="central-bank",
                kind=AuxiliaryFeatureKind.MACRO,
                symbol=None,
                revision_id=str(i),
            )
        )
    return PointInTimeFeatureStore(
        tuple(observations),
        PointInTimeFeatureManifest(
            source="synthetic",
            point_in_time=True,
            revision_aware=True,
        ),
    )


def test_training_can_use_only_gated_point_in_time_auxiliary_features():
    bars, dates = _bars()
    metadata = DatasetMetadata("aux-test", "synthetic", DataGrade.RESEARCH_GRADE)
    run = run_training_research(
        bars,
        metadata,
        model_configs=(ModelConfig("ridge", {"alpha": 1.0}, seed=7),),
        horizon=5,
        train_periods=100,
        test_periods=20,
        auxiliary_store=_store(dates),
        auxiliary_feature_names=("policy_rate",),
        auxiliary_min_coverage=0.95,
    )

    assert run.auxiliary_features == ("aux__policy_rate",)
    assert len(run.leaderboard) == 1
    assert run.leaderboard[0].predictions is not None
    assert run.leaderboard[0].predictions.notna().sum() > 0


def test_training_rejects_non_point_in_time_auxiliary_manifest():
    bars, dates = _bars()
    safe = _store(dates)
    unsafe = PointInTimeFeatureStore(
        safe.observations,
        PointInTimeFeatureManifest(
            source="synthetic",
            point_in_time=False,
            revision_aware=True,
        ),
    )
    metadata = DatasetMetadata("aux-test", "synthetic", DataGrade.RESEARCH_GRADE)
    try:
        run_training_research(
            bars,
            metadata,
            model_configs=(ModelConfig("ridge", seed=7),),
            horizon=5,
            train_periods=100,
            test_periods=20,
            auxiliary_store=unsafe,
            auxiliary_feature_names=("policy_rate",),
        )
    except ValueError as exc:
        assert "auxiliary point-in-time feature gate failed" in str(exc)
    else:
        raise AssertionError("expected unsafe auxiliary feature store to be rejected")
