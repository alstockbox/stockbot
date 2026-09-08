from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from stockbot.arena.experiments import ModelExperimentResult
from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.ml.models import ModelConfig
from stockbot.research.champion import ChampionState, JsonChampionStore, make_champion_state
from stockbot.research.drift import DriftReport, evaluate_return_drift
from stockbot.research.ensemble import EnsembleReport, build_horizon_ensemble
from stockbot.research.gates import GateDecision, ResearchGateCriteria, evaluate_research_gate
from stockbot.research.holdout import HoldoutConfig, HoldoutReport, evaluate_blind_holdout, split_research_holdout
from stockbot.research.memory import JsonlExperimentMemory, experiment_id, make_record
from stockbot.research.multiple_testing import DiscoveryResult, evaluate_discoveries
from stockbot.research.objective import ObjectiveWeights, risk_adjusted_objective
from stockbot.research.population import ModelPopulationConfig, generate_model_population
from stockbot.research.regime_eval import RegimePerformanceReport, build_market_regime_series, evaluate_regime_performance
from stockbot.research.stress import StressReport, evaluate_stress_suite
from stockbot.research.training_pipeline import run_training_research


@dataclass(frozen=True)
class ResearchFactoryConfig:
    """Configuration for large-scale autonomous challenger search."""

    horizons: tuple[int, ...] = (1, 5, 20)
    population: ModelPopulationConfig = field(default_factory=ModelPopulationConfig)
    gates: ResearchGateCriteria = field(default_factory=ResearchGateCriteria)
    objective: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    holdout: HoldoutConfig = field(default_factory=HoldoutConfig)
    holdout_top_k: int = 12
    require_holdout: bool = True
    promotion_margin: float = 0.02
    max_workers: int = 4
    max_fdr_q_value: float = 0.20
    require_statistical_discovery: bool = True
    min_regime_score: float = 0.20
    min_regime_coverage: float = 1.0 / 3.0
    reject_drifted_candidates: bool = True
    ensemble_temperature: float = 0.75

    def __post_init__(self) -> None:
        if not self.horizons or any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")
        if self.promotion_margin < 0:
            raise ValueError("promotion_margin must be non-negative")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.holdout_top_k <= 0:
            raise ValueError("holdout_top_k must be positive")
        if not 0.0 < self.max_fdr_q_value < 1.0:
            raise ValueError("max_fdr_q_value must be in (0,1)")
        if not 0.0 <= self.min_regime_score <= 1.0:
            raise ValueError("min_regime_score must be in [0,1]")
        if not 0.0 <= self.min_regime_coverage <= 1.0:
            raise ValueError("min_regime_coverage must be in [0,1]")
        if self.ensemble_temperature <= 0.0:
            raise ValueError("ensemble_temperature must be positive")


@dataclass(frozen=True)
class FactoryCandidate:
    experiment_id: str
    horizon: int
    model_name: str
    model_params: dict[str, Any]
    seed: int
    factory_score: float
    selection_score: float
    promotion_score: float
    base_score: float
    metrics: dict[str, float]
    robustness: float
    oos_coverage: float
    stress_score: float
    stress_report: StressReport | None
    regime_report: RegimePerformanceReport | None
    discovery: DiscoveryResult | None
    drift_report: DriftReport | None
    holdout_report: HoldoutReport | None
    gate: GateDecision
    result: ModelExperimentResult


@dataclass(frozen=True)
class FactoryReport:
    candidates: tuple[FactoryCandidate, ...]
    champion_candidate: FactoryCandidate | None
    active_champion: ChampionState | None
    promotion_occurred: bool
    horizon_champions: dict[int, FactoryCandidate]
    ensemble_report: EnsembleReport | None
    data_grade: DataGrade
    experiments_run: int
    candidates_passed: int
    holdout_evaluated: int
    holdout_start: pd.Timestamp | None


class ResearchFactory:
    """Autonomous multi-horizon research factory with layered anti-overfit controls.

    Hyperparameters are selected only on the research partition. Candidates are then
    filtered by transaction-cost-aware OOS performance, adversarial stress, market
    regime robustness, false-discovery control and recent-return drift before the
    strongest models may touch the blind final holdout. Horizon winners are also
    blended into a diversified research ensemble for meta-level evaluation.
    """

    def __init__(
        self,
        config: ResearchFactoryConfig | None = None,
        *,
        memory: JsonlExperimentMemory | None = None,
        champion_store: JsonChampionStore | None = None,
    ) -> None:
        self.config = config or ResearchFactoryConfig()
        self.memory = memory
        self.champion_store = champion_store

    @staticmethod
    def _model_from_result(result: ModelExperimentResult) -> ModelConfig:
        return ModelConfig(
            result.artifact.model_name,
            params=dict(result.artifact.model_params),
            seed=int(result.artifact.seed),
        )

    @staticmethod
    def _stress(result: ModelExperimentResult) -> StressReport | None:
        if result.net_returns is None or result.turnover_series is None:
            return None
        if len(result.net_returns) == 0:
            return None
        return evaluate_stress_suite(
            result.net_returns,
            turnover=result.turnover_series,
        )

    def _candidate(
        self,
        result: ModelExperimentResult,
        metadata: DatasetMetadata,
        *,
        horizon: int,
    ) -> FactoryCandidate:
        model = self._model_from_result(result)
        stress_report = self._stress(result)
        stress_score = stress_report.score if stress_report is not None else 0.0
        factory_score = risk_adjusted_objective(
            result.metrics,
            robustness=result.robustness,
            oos_coverage=result.oos_coverage,
            stress_score=stress_score,
            weights=self.config.objective,
        )
        gate = evaluate_research_gate(
            result,
            metadata,
            factory_score=factory_score,
            stress_score=stress_score,
            criteria=self.config.gates,
        )
        exp_id = experiment_id(
            model,
            dataset_fingerprint=result.artifact.dataset_fingerprint,
            horizon=horizon,
        )
        candidate = FactoryCandidate(
            experiment_id=exp_id,
            horizon=int(horizon),
            model_name=model.name,
            model_params=dict(model.params),
            seed=int(model.seed),
            factory_score=float(factory_score),
            selection_score=float(factory_score),
            promotion_score=float(factory_score),
            base_score=float(result.score),
            metrics={key: float(value) for key, value in result.metrics.items()},
            robustness=float(result.robustness),
            oos_coverage=float(result.oos_coverage),
            stress_score=float(stress_score),
            stress_report=stress_report,
            regime_report=None,
            discovery=None,
            drift_report=None,
            holdout_report=None,
            gate=gate,
            result=result,
        )

        if self.memory is not None and not self.memory.seen(exp_id):
            self.memory.append(
                make_record(
                    model,
                    dataset_fingerprint=result.artifact.dataset_fingerprint,
                    horizon=horizon,
                    factory_score=factory_score,
                    base_score=result.score,
                    robustness=result.robustness,
                    oos_coverage=result.oos_coverage,
                    metrics=result.metrics,
                    passed_gates=gate.passed,
                    rejection_reasons=gate.reasons,
                    stress_score=stress_score,
                )
            )
        return candidate

    def _enrich_candidates(
        self,
        candidates: list[FactoryCandidate],
        research_bars: pd.DataFrame,
    ) -> list[FactoryCandidate]:
        regime_series = build_market_regime_series(research_bars)
        returns_by_id = {
            candidate.experiment_id: candidate.result.net_returns
            for candidate in candidates
            if candidate.result.net_returns is not None and len(candidate.result.net_returns) > 0
        }
        discoveries = evaluate_discoveries(
            returns_by_id,
            max_q_value=self.config.max_fdr_q_value,
        ) if returns_by_id else {}

        enriched: list[FactoryCandidate] = []
        for candidate in candidates:
            returns = candidate.result.net_returns
            if returns is None or len(returns) == 0:
                enriched.append(candidate)
                continue
            regime_report = evaluate_regime_performance(returns, regime_series)
            drift_report = evaluate_return_drift(returns)
            discovery = discoveries.get(candidate.experiment_id)
            discovery_confidence = discovery.confidence if discovery is not None else 0.0
            selection_score = (
                candidate.factory_score
                + 1.00 * regime_report.score
                + 0.75 * discovery_confidence
                + 0.50 * drift_report.score
            )
            enriched.append(
                replace(
                    candidate,
                    selection_score=float(selection_score),
                    promotion_score=float(selection_score),
                    regime_report=regime_report,
                    discovery=discovery,
                    drift_report=drift_report,
                )
            )
        return enriched

    def _pre_holdout_eligible(self, candidate: FactoryCandidate) -> bool:
        if not candidate.gate.passed:
            return False
        if candidate.regime_report is None:
            return False
        if candidate.regime_report.score < self.config.min_regime_score:
            return False
        if candidate.regime_report.coverage < self.config.min_regime_coverage:
            return False
        if self.config.require_statistical_discovery:
            if candidate.discovery is None or not candidate.discovery.significant:
                return False
        if self.config.reject_drifted_candidates:
            if candidate.drift_report is None or candidate.drift_report.degraded:
                return False
        return True

    def _apply_blind_holdout(
        self,
        bars: pd.DataFrame,
        candidates: list[FactoryCandidate],
        *,
        holdout_start: pd.Timestamp,
    ) -> tuple[list[FactoryCandidate], int]:
        selected: list[FactoryCandidate] = []
        for horizon in self.config.horizons:
            horizon_passed = [
                candidate
                for candidate in candidates
                if self._pre_holdout_eligible(candidate) and candidate.horizon == horizon
            ]
            horizon_passed.sort(key=lambda row: row.selection_score, reverse=True)
            selected.extend(horizon_passed[: self.config.holdout_top_k])

        replacements: dict[str, FactoryCandidate] = {}
        for candidate in selected:
            model = ModelConfig(
                candidate.model_name,
                params=dict(candidate.model_params),
                seed=candidate.seed,
            )
            report = evaluate_blind_holdout(
                bars,
                model,
                horizon=candidate.horizon,
                holdout_start=holdout_start,
                config=self.config.holdout,
                objective=self.config.objective,
            )
            promotion_score = 0.70 * candidate.selection_score + 0.30 * report.score
            replacements[candidate.experiment_id] = replace(
                candidate,
                holdout_report=report,
                promotion_score=float(promotion_score),
            )

        updated = [replacements.get(candidate.experiment_id, candidate) for candidate in candidates]
        return updated, len(selected)

    @staticmethod
    def _state_from_candidate(candidate: FactoryCandidate) -> ChampionState:
        holdout_score = (
            candidate.holdout_report.score
            if candidate.holdout_report is not None
            else candidate.promotion_score
        )
        return make_champion_state(
            experiment_id=candidate.experiment_id,
            horizon=candidate.horizon,
            model_name=candidate.model_name,
            model_params=candidate.model_params,
            seed=candidate.seed,
            dataset_fingerprint=candidate.result.artifact.dataset_fingerprint,
            factory_score=candidate.factory_score,
            promotion_score=candidate.promotion_score,
            holdout_score=holdout_score,
        )

    def _build_ensemble(
        self,
        horizon_champions: dict[int, FactoryCandidate],
        research_bars: pd.DataFrame,
    ) -> EnsembleReport | None:
        usable = {
            candidate.experiment_id: candidate
            for candidate in horizon_champions.values()
            if candidate.result.net_returns is not None and len(candidate.result.net_returns) > 0
        }
        if not usable:
            return None
        regime_series = build_market_regime_series(research_bars)
        return build_horizon_ensemble(
            {item: candidate.result.net_returns for item, candidate in usable.items()},
            {item: candidate.promotion_score for item, candidate in usable.items()},
            regime_series=regime_series,
            temperature=self.config.ensemble_temperature,
        )

    def run(self, bars: pd.DataFrame, metadata: DatasetMetadata) -> FactoryReport:
        incumbent = self.champion_store.load() if self.champion_store is not None else None

        holdout_start: pd.Timestamp | None = None
        research_bars = bars
        if self.config.require_holdout:
            research_bars, holdout_start = split_research_holdout(bars, self.config.holdout)

        population = generate_model_population(self.config.population)
        candidates: list[FactoryCandidate] = []

        for horizon in self.config.horizons:
            training_run = run_training_research(
                research_bars,
                metadata,
                model_configs=population,
                horizon=horizon,
                max_workers=self.config.max_workers,
            )
            candidates.extend(
                self._candidate(result, metadata, horizon=horizon)
                for result in training_run.leaderboard
            )

        candidates = self._enrich_candidates(candidates, research_bars)
        candidates.sort(key=lambda row: row.selection_score, reverse=True)

        holdout_evaluated = 0
        if holdout_start is not None:
            candidates, holdout_evaluated = self._apply_blind_holdout(
                bars,
                candidates,
                holdout_start=holdout_start,
            )

        if holdout_start is None:
            promotion_pool = [candidate for candidate in candidates if self._pre_holdout_eligible(candidate)]
        else:
            promotion_pool = [
                candidate
                for candidate in candidates
                if self._pre_holdout_eligible(candidate)
                and candidate.holdout_report is not None
                and candidate.holdout_report.passed
            ]
        promotion_pool.sort(key=lambda row: row.promotion_score, reverse=True)

        horizon_champions: dict[int, FactoryCandidate] = {}
        for horizon in self.config.horizons:
            winner = next((row for row in promotion_pool if row.horizon == horizon), None)
            if winner is not None:
                horizon_champions[horizon] = winner

        ensemble_report = self._build_ensemble(horizon_champions, research_bars)

        challenger = promotion_pool[0] if promotion_pool else None
        promoted = False
        active_champion = incumbent
        champion_candidate: FactoryCandidate | None = None

        if challenger is not None:
            qualifies_against_incumbent = (
                incumbent is None
                or challenger.promotion_score
                >= incumbent.promotion_score + self.config.promotion_margin
            )
            if qualifies_against_incumbent:
                champion_candidate = challenger
                active_champion = self._state_from_candidate(challenger)
                promoted = True
                if self.champion_store is not None:
                    self.champion_store.save(active_champion)

        return FactoryReport(
            candidates=tuple(candidates),
            champion_candidate=champion_candidate,
            active_champion=active_champion,
            promotion_occurred=promoted,
            horizon_champions=horizon_champions,
            ensemble_report=ensemble_report,
            data_grade=metadata.grade,
            experiments_run=len(candidates),
            candidates_passed=len(promotion_pool),
            holdout_evaluated=holdout_evaluated,
            holdout_start=holdout_start,
        )
