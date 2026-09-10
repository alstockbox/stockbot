import numpy as np
import pandas as pd

from stockbot.data.point_in_time_features import (
    AuxiliaryFeatureKind,
    PointInTimeFeatureManifest,
    PointInTimeFeatureObservation,
    PointInTimeFeatureStore,
)
from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.research.factory import ResearchFactory, ResearchFactoryConfig
from stockbot.research.gates import ResearchGateCriteria
from stockbot.research.holdout import HoldoutConfig
from stockbot.research.population import ModelPopulationConfig


def _bars():
    rng = np.random.default_rng(20260908)
    dates = pd.date_range("2024-01-02", periods=240, freq="B", tz="UTC")
    rows = []
    market = rng.normal(0.0002, 0.005, len(dates))
    for j, symbol in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")):
        local = market + 0.0005 * np.sin(np.arange(len(dates)) / (9.0 + j)) + rng.normal(0.0, 0.004, len(dates))
        close = 100.0 * np.exp(np.cumsum(local))
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": float(close[i]),
                    "high": float(close[i] * 1.008),
                    "low": float(close[i] * 0.992),
                    "close": float(close[i]),
                    "volume": int(800_000 + 50_000 * j + 3_000 * (i % 23)),
                }
            )
    return pd.DataFrame(rows), dates


def _store(dates):
    observations = tuple(
        PointInTimeFeatureObservation(
            feature_name="policy_rate",
            value=2.0 + 0.05 * i,
            observation_time=dt.isoformat(),
            available_time=dt.isoformat(),
            source="central-bank",
            kind=AuxiliaryFeatureKind.MACRO,
            revision_id=str(i),
        )
        for i, dt in enumerate(dates[::20])
    )
    return PointInTimeFeatureStore(
        observations,
        PointInTimeFeatureManifest(
            source="synthetic",
            point_in_time=True,
            revision_aware=True,
        ),
    )


def _config():
    gates = ResearchGateCriteria(
        min_oos_coverage=0.10,
        min_robustness=0.0,
        min_stress_score=0.0,
        max_drawdown=1.0,
        max_cvar_95=1.0,
        max_turnover=1_000_000.0,
        max_instability=1.0,
        max_concentration=1.0,
        min_factory_score=-1_000_000.0,
        require_research_grade=True,
    )
    holdout = HoldoutConfig(
        fraction=0.15,
        min_holdout_periods=30,
        min_research_periods=140,
        min_prediction_coverage=0.20,
        max_drawdown=1.0,
        min_stress_score=0.0,
        min_score=-1_000_000.0,
    )
    return ResearchFactoryConfig(
        horizons=(5,),
        population=ModelPopulationConfig(max_candidates=1, seeds=(7,)),
        gates=gates,
        holdout=holdout,
        holdout_top_k=1,
        promotion_margin=0.0,
        max_workers=1,
        require_statistical_discovery=False,
        min_regime_score=0.0,
        min_regime_coverage=0.0,
        reject_drifted_candidates=False,
        require_paper_readiness=False,
    )


def test_v2_auxiliary_features_replay_through_blind_holdout_and_change_identity():
    bars, dates = _bars()
    metadata = DatasetMetadata("factory-aux", "synthetic", DataGrade.RESEARCH_GRADE)
    auxiliary = ResearchFactory(_config()).run(
        bars,
        metadata,
        auxiliary_store=_store(dates),
        auxiliary_feature_names=("policy_rate",),
        auxiliary_min_coverage=0.95,
    )
    price_only = ResearchFactory(_config()).run(bars, metadata)

    assert auxiliary.auxiliary_features == ("aux__policy_rate",)
    assert auxiliary.holdout_evaluated == 1
    candidate = auxiliary.candidates[0]
    assert "aux__policy_rate" in candidate.result.artifact.feature_names
    assert candidate.holdout_report is not None
    assert candidate.result.artifact.dataset_fingerprint != price_only.candidates[0].result.artifact.dataset_fingerprint
