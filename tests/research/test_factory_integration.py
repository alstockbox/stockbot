import numpy as np
import pandas as pd

from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.research.champion import JsonChampionStore
from stockbot.research.factory import ResearchFactory, ResearchFactoryConfig
from stockbot.research.gates import ResearchGateCriteria
from stockbot.research.holdout import HoldoutConfig
from stockbot.research.memory import JsonlExperimentMemory
from stockbot.research.population import ModelPopulationConfig


def _research_bars() -> pd.DataFrame:
    rng = np.random.default_rng(20260908)
    dates = pd.date_range("2024-01-02", periods=240, freq="B", tz="UTC")
    rows = []
    market = rng.normal(0.00025, 0.006, len(dates))
    for j, symbol in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]):
        alpha = 0.00004 * j
        cyclical = 0.0008 * np.sin(np.arange(len(dates)) / (9.0 + j))
        noise = rng.normal(0.0, 0.005 + 0.0003 * j, len(dates))
        close = 100.0 * np.exp(np.cumsum(market + alpha + cyclical + noise))
        volume = rng.integers(800_000, 3_000_000, len(dates))
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": float(close[i] * (1.0 + rng.normal(0.0, 0.001))),
                    "high": float(close[i] * 1.009),
                    "low": float(close[i] * 0.991),
                    "close": float(close[i]),
                    "volume": int(volume[i]),
                }
            )
    return pd.DataFrame(rows)


def test_research_factory_runs_population_stress_holdout_and_champion(tmp_path):
    relaxed_gates = ResearchGateCriteria(
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
    config = ResearchFactoryConfig(
        horizons=(5,),
        population=ModelPopulationConfig(max_candidates=5, seeds=(7,)),
        gates=relaxed_gates,
        holdout=holdout,
        holdout_top_k=2,
        promotion_margin=0.0,
        max_workers=2,
        require_statistical_discovery=False,
        min_regime_score=0.0,
        min_regime_coverage=0.0,
        reject_drifted_candidates=False,
        require_paper_readiness=False,
    )
    memory = JsonlExperimentMemory(tmp_path / "experiments.jsonl")
    champion_store = JsonChampionStore(tmp_path / "champion.json")
    factory = ResearchFactory(config, memory=memory, champion_store=champion_store)
    metadata = DatasetMetadata(
        name="factory-integration",
        source="synthetic-test",
        grade=DataGrade.RESEARCH_GRADE,
    )

    report = factory.run(_research_bars(), metadata)

    assert report.experiments_run == 5
    assert len(report.candidates) == 5
    assert all(candidate.stress_report is not None for candidate in report.candidates)
    assert all(candidate.regime_report is not None for candidate in report.candidates)
    assert all(candidate.discovery is not None for candidate in report.candidates)
    assert all(candidate.drift_report is not None for candidate in report.candidates)
    assert all(candidate.paper_readiness is not None for candidate in report.candidates)
    assert report.holdout_start is not None
    assert report.holdout_evaluated == 2
    assert report.candidates_passed >= 1
    assert report.promotion_occurred
    assert report.champion_candidate is not None
    assert report.active_champion is not None
    assert report.ensemble_report is not None
    assert champion_store.load() == report.active_champion
    assert len(memory.records()) == 5
