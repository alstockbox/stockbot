from stockbot.research.champion import ChampionState, JsonChampionStore
from stockbot.research.drift import DriftReport, evaluate_return_drift
from stockbot.research.ensemble import EnsembleReport, build_horizon_ensemble
from stockbot.research.factory import FactoryReport, ResearchFactory, ResearchFactoryConfig
from stockbot.research.failures import mine_hard_negatives
from stockbot.research.gates import GateDecision, ResearchGateCriteria, evaluate_research_gate
from stockbot.research.holdout import HoldoutConfig, HoldoutReport, evaluate_blind_holdout, split_research_holdout
from stockbot.research.jobs import ResearchJobManifest, ResearchRunSummary, make_job_manifest, make_run_summary, write_json_record
from stockbot.research.market_training import run_snapshot_factory, train_snapshot
from stockbot.research.memory import ExperimentRecord, JsonlExperimentMemory
from stockbot.research.multiple_testing import DiscoveryResult, benjamini_hochberg, evaluate_discoveries
from stockbot.research.objective import ObjectiveWeights, risk_adjusted_objective
from stockbot.research.population import ModelPopulationConfig, generate_model_population
from stockbot.research.readiness import PaperReadinessReport, evaluate_paper_readiness
from stockbot.research.regime_eval import RegimePerformanceReport, build_market_regime_series, evaluate_regime_performance
from stockbot.research.stress import StressReport, StressScenario, default_stress_suite, evaluate_stress_suite

__all__ = [
    "ChampionState",
    "DiscoveryResult",
    "DriftReport",
    "EnsembleReport",
    "ExperimentRecord",
    "FactoryReport",
    "GateDecision",
    "HoldoutConfig",
    "HoldoutReport",
    "JsonChampionStore",
    "JsonlExperimentMemory",
    "ModelPopulationConfig",
    "ObjectiveWeights",
    "PaperReadinessReport",
    "RegimePerformanceReport",
    "ResearchFactory",
    "ResearchFactoryConfig",
    "ResearchGateCriteria",
    "ResearchJobManifest",
    "ResearchRunSummary",
    "StressReport",
    "StressScenario",
    "benjamini_hochberg",
    "build_horizon_ensemble",
    "build_market_regime_series",
    "default_stress_suite",
    "evaluate_blind_holdout",
    "evaluate_discoveries",
    "evaluate_paper_readiness",
    "evaluate_regime_performance",
    "evaluate_research_gate",
    "evaluate_return_drift",
    "evaluate_stress_suite",
    "generate_model_population",
    "make_job_manifest",
    "make_run_summary",
    "mine_hard_negatives",
    "risk_adjusted_objective",
    "run_snapshot_factory",
    "split_research_holdout",
    "train_snapshot",
    "write_json_record",
]
