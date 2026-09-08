from stockbot.research.adaptive_population import AdaptivePopulationConfig, generate_adaptive_population
from stockbot.research.champion import ChampionState, JsonChampionStore
from stockbot.research.deep_diagnostics import CandidateDeepDiagnostics, DeepResearchDiagnostics, compact_deep_diagnostics, run_deep_research_diagnostics
from stockbot.research.drift import DriftReport, evaluate_return_drift
from stockbot.research.ensemble import EnsembleReport, build_horizon_ensemble
from stockbot.research.factory import FactoryReport, ResearchFactory, ResearchFactoryConfig
from stockbot.research.failures import mine_hard_negatives
from stockbot.research.feature_ablation import FeatureAblationReport, FeatureAblationResult, evaluate_feature_group_ablation
from stockbot.research.gates import GateDecision, ResearchGateCriteria, evaluate_research_gate
from stockbot.research.holdout import HoldoutConfig, HoldoutReport, evaluate_blind_holdout, split_research_holdout
from stockbot.research.jobs import ResearchJobManifest, ResearchRunSummary, make_job_manifest, make_run_summary, write_json_payload, write_json_record
from stockbot.research.market_training import run_snapshot_deep_diagnostics, run_snapshot_factory, run_snapshot_regime_specialists, train_snapshot
from stockbot.research.memory import ExperimentRecord, JsonlExperimentMemory
from stockbot.research.multiple_testing import DiscoveryResult, benjamini_hochberg, evaluate_discoveries
from stockbot.research.objective import ObjectiveWeights, risk_adjusted_objective
from stockbot.research.policy_search import PolicyArenaReport, PolicyResult, PortfolioPolicy, evaluate_policy_arena
from stockbot.research.population import ModelPopulationConfig, generate_model_population
from stockbot.research.readiness import PaperReadinessReport, evaluate_paper_readiness
from stockbot.research.regime_eval import RegimePerformanceReport, build_market_regime_series, evaluate_regime_performance
from stockbot.research.regime_router import RegimeRouterReport, build_regime_router
from stockbot.research.regime_specialists import RegimeSpecialistResult, run_regime_specialist, select_best_regime_specialists
from stockbot.research.specialist_pipeline import RegimeSpecialistDiagnostics, run_regime_specialist_diagnostics, select_specialist_candidate_pairs
from stockbot.research.stress import StressReport, StressScenario, default_stress_suite, evaluate_stress_suite
from stockbot.research.window_robustness import WindowResult, WindowRobustnessReport, evaluate_training_window_robustness

__all__ = [
    "AdaptivePopulationConfig",
    "CandidateDeepDiagnostics",
    "ChampionState",
    "DeepResearchDiagnostics",
    "DiscoveryResult",
    "DriftReport",
    "EnsembleReport",
    "ExperimentRecord",
    "FactoryReport",
    "FeatureAblationReport",
    "FeatureAblationResult",
    "GateDecision",
    "HoldoutConfig",
    "HoldoutReport",
    "JsonChampionStore",
    "JsonlExperimentMemory",
    "ModelPopulationConfig",
    "ObjectiveWeights",
    "PaperReadinessReport",
    "PolicyArenaReport",
    "PolicyResult",
    "PortfolioPolicy",
    "RegimePerformanceReport",
    "RegimeRouterReport",
    "RegimeSpecialistDiagnostics",
    "RegimeSpecialistResult",
    "ResearchFactory",
    "ResearchFactoryConfig",
    "ResearchGateCriteria",
    "ResearchJobManifest",
    "ResearchRunSummary",
    "StressReport",
    "StressScenario",
    "WindowResult",
    "WindowRobustnessReport",
    "benjamini_hochberg",
    "build_horizon_ensemble",
    "build_market_regime_series",
    "build_regime_router",
    "compact_deep_diagnostics",
    "default_stress_suite",
    "evaluate_blind_holdout",
    "evaluate_discoveries",
    "evaluate_feature_group_ablation",
    "evaluate_paper_readiness",
    "evaluate_policy_arena",
    "evaluate_regime_performance",
    "evaluate_research_gate",
    "evaluate_return_drift",
    "evaluate_stress_suite",
    "evaluate_training_window_robustness",
    "generate_adaptive_population",
    "generate_model_population",
    "make_job_manifest",
    "make_run_summary",
    "mine_hard_negatives",
    "risk_adjusted_objective",
    "run_deep_research_diagnostics",
    "run_regime_specialist",
    "run_regime_specialist_diagnostics",
    "run_snapshot_deep_diagnostics",
    "run_snapshot_factory",
    "run_snapshot_regime_specialists",
    "select_best_regime_specialists",
    "select_specialist_candidate_pairs",
    "split_research_holdout",
    "train_snapshot",
    "write_json_payload",
    "write_json_record",
]
