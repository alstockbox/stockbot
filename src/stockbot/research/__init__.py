from stockbot.research.adaptive_population import AdaptivePopulationConfig, generate_adaptive_population
from stockbot.research.auxiliary_ablation import AuxiliaryAblationReport, AuxiliaryAblationResult, evaluate_auxiliary_feature_ablation
from stockbot.research.bootstrap_uncertainty import BootstrapInterval, BootstrapUncertaintyReport, evaluate_block_bootstrap_uncertainty
from stockbot.research.capacity_curve import CapacityCurveReport, CapacityPoint, evaluate_capacity_curve
from stockbot.research.champion import ChampionState, JsonChampionStore
from stockbot.research.deep_diagnostics import CandidateDeepDiagnostics, DeepResearchDiagnostics, compact_deep_diagnostics, run_deep_research_diagnostics
from stockbot.research.deep_feedback import (
    DeepResearchFinding,
    JsonlDeepResearchMemory,
    build_research_cycle_id,
    make_deep_findings,
    rank_adaptive_parent_records,
)
from stockbot.research.drift import DriftReport, evaluate_return_drift
from stockbot.research.ensemble import EnsembleReport, build_horizon_ensemble
from stockbot.research.factor_exposure import FactorExposureReport, build_internal_factor_returns, evaluate_factor_exposure
from stockbot.research.factory import FactoryReport, ResearchFactory, ResearchFactoryConfig
from stockbot.research.failures import mine_hard_negatives
from stockbot.research.feature_ablation import FeatureAblationReport, FeatureAblationResult, evaluate_feature_group_ablation
from stockbot.research.gates import GateDecision, ResearchGateCriteria, evaluate_research_gate
from stockbot.research.holdout import HoldoutConfig, HoldoutReport, evaluate_blind_holdout, split_research_holdout
from stockbot.research.jobs import ResearchJobManifest, ResearchRunSummary, make_job_manifest, make_run_summary, write_json_payload, write_json_record
from stockbot.research.market_training import run_snapshot_deep_diagnostics, run_snapshot_factory, run_snapshot_regime_specialists, train_snapshot
from stockbot.research.memory import ExperimentRecord, JsonlExperimentMemory
from stockbot.research.multiple_testing import DiscoveryResult, benjamini_hochberg, evaluate_discoveries
from stockbot.research.neutralization import NeutralizationConfig, NeutralizationReport, evaluate_sector_factor_neutralization
from stockbot.research.objective import ObjectiveWeights, risk_adjusted_objective
from stockbot.research.policy_search import PolicyArenaReport, PolicyResult, PortfolioPolicy, evaluate_policy_arena
from stockbot.research.population import ModelPopulationConfig, generate_model_population
from stockbot.research.quarantine import (
    QuarantineConfig,
    QuarantineManifest,
    QuarantineSplit,
    assert_same_quarantine_boundary,
    load_quarantine_manifest,
    persist_sealed_quarantine,
    split_sealed_quarantine,
)
from stockbot.research.quarantine_audit import (
    FrozenStrategySpec,
    QuarantineAuditRecord,
    QuarantineAuditResult,
    freeze_strategy_spec,
    run_single_quarantine_audit,
)
from stockbot.research.readiness import PaperReadinessReport, evaluate_paper_readiness
from stockbot.research.regime_eval import RegimePerformanceReport, build_market_regime_series, evaluate_regime_performance
from stockbot.research.regime_router import RegimeRouterReport, build_regime_router
from stockbot.research.regime_specialists import RegimeSpecialistResult, run_regime_specialist, select_best_regime_specialists
from stockbot.research.sector_allocator import SectorCapAllocationReport, apply_point_in_time_sector_cap, evaluate_sector_cap_challenger
from stockbot.research.sector_exposure import SectorPortfolioExposureReport, evaluate_sector_portfolio_exposure
from stockbot.research.specialist_pipeline import RegimeSpecialistDiagnostics, run_regime_specialist_diagnostics, select_specialist_candidate_pairs
from stockbot.research.stacking import StackingReport, evaluate_oos_stacking
from stockbot.research.stress import StressReport, StressScenario, default_stress_suite, evaluate_stress_suite
from stockbot.research.window_robustness import WindowResult, WindowRobustnessReport, evaluate_training_window_robustness

__all__ = [
    "AdaptivePopulationConfig",
    "AuxiliaryAblationReport",
    "AuxiliaryAblationResult",
    "BootstrapInterval",
    "BootstrapUncertaintyReport",
    "CandidateDeepDiagnostics",
    "CapacityCurveReport",
    "CapacityPoint",
    "ChampionState",
    "DeepResearchDiagnostics",
    "DeepResearchFinding",
    "DiscoveryResult",
    "DriftReport",
    "EnsembleReport",
    "ExperimentRecord",
    "FactorExposureReport",
    "FactoryReport",
    "FeatureAblationReport",
    "FeatureAblationResult",
    "FrozenStrategySpec",
    "GateDecision",
    "HoldoutConfig",
    "HoldoutReport",
    "JsonChampionStore",
    "JsonlDeepResearchMemory",
    "JsonlExperimentMemory",
    "ModelPopulationConfig",
    "NeutralizationConfig",
    "NeutralizationReport",
    "ObjectiveWeights",
    "PaperReadinessReport",
    "PolicyArenaReport",
    "PolicyResult",
    "PortfolioPolicy",
    "QuarantineAuditRecord",
    "QuarantineAuditResult",
    "QuarantineConfig",
    "QuarantineManifest",
    "QuarantineSplit",
    "RegimePerformanceReport",
    "RegimeRouterReport",
    "RegimeSpecialistDiagnostics",
    "RegimeSpecialistResult",
    "ResearchFactory",
    "ResearchFactoryConfig",
    "ResearchGateCriteria",
    "ResearchJobManifest",
    "ResearchRunSummary",
    "SectorCapAllocationReport",
    "SectorPortfolioExposureReport",
    "StackingReport",
    "StressReport",
    "StressScenario",
    "WindowResult",
    "WindowRobustnessReport",
    "apply_point_in_time_sector_cap",
    "assert_same_quarantine_boundary",
    "benjamini_hochberg",
    "build_horizon_ensemble",
    "build_internal_factor_returns",
    "build_market_regime_series",
    "build_regime_router",
    "build_research_cycle_id",
    "compact_deep_diagnostics",
    "default_stress_suite",
    "evaluate_auxiliary_feature_ablation",
    "evaluate_blind_holdout",
    "evaluate_block_bootstrap_uncertainty",
    "evaluate_capacity_curve",
    "evaluate_discoveries",
    "evaluate_factor_exposure",
    "evaluate_feature_group_ablation",
    "evaluate_oos_stacking",
    "evaluate_paper_readiness",
    "evaluate_policy_arena",
    "evaluate_regime_performance",
    "evaluate_research_gate",
    "evaluate_return_drift",
    "evaluate_sector_cap_challenger",
    "evaluate_sector_factor_neutralization",
    "evaluate_sector_portfolio_exposure",
    "evaluate_stress_suite",
    "evaluate_training_window_robustness",
    "freeze_strategy_spec",
    "generate_adaptive_population",
    "generate_model_population",
    "load_quarantine_manifest",
    "make_deep_findings",
    "make_job_manifest",
    "make_run_summary",
    "mine_hard_negatives",
    "persist_sealed_quarantine",
    "rank_adaptive_parent_records",
    "risk_adjusted_objective",
    "run_deep_research_diagnostics",
    "run_regime_specialist",
    "run_regime_specialist_diagnostics",
    "run_single_quarantine_audit",
    "run_snapshot_deep_diagnostics",
    "run_snapshot_factory",
    "run_snapshot_regime_specialists",
    "select_best_regime_specialists",
    "select_specialist_candidate_pairs",
    "split_research_holdout",
    "split_sealed_quarantine",
    "train_snapshot",
    "write_json_payload",
    "write_json_record",
]
