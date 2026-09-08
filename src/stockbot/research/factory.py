from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from stockbot.arena.experiments import ModelExperimentResult
from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.ml.models import ModelConfig
from stockbot.research.gates import GateDecision, ResearchGateCriteria, evaluate_research_gate
from stockbot.research.holdout import HoldoutConfig, HoldoutReport, evaluate_blind_holdout, split_research_holdout
from stockbot.research.memory import JsonlExperimentMemory, experiment_id, make_record
from stockbot.research.objective import ObjectiveWeights, risk_adjusted_objective
from stockbot.research.population import ModelPopulationConfig, generate_model_population
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


@dataclass(frozen=True)
class FactoryCandidate:
    experiment_id: str
    horizon: int
    model_name: str
    model_params: dict[str, Any]
    seed: int
    factory_score: float
    promotion_score: float
    base_score: float
    metrics: dict[str, float]
    robustness: float
    oos_coverage: float
    stress_score: float
    stress_report: StressReport | None
    holdout_report: HoldoutReport | None
    gate: GateDecision
    result: ModelExperimentResult


@dataclass(frozen=True)
class FactoryReport:
    candidates: tuple[FactoryCandidate, ...]
    champion_candidate: FactoryCandidate | None
    horizon_champions: dict[int, FactoryCandidate]
    data_grade: DataGrade
    experiments_run: int
    candidates_passed: int
    holdout_evaluated: int
    holdout_start: pd.Timestamp | None


class ResearchFactory:
    """Run broad, multi-horizon ML research and promote only robust OOS candidates.

    Hyperparameters are selected only on the research partition. The final time block
    is reserved before the search starts and is exposed only to the strongest gated
    candidates. Promotion therefore requires both repeated purged walk-forward edge
    and survival on an untouched blind holdout.
    """

    def __init__(
        self,
        config: ResearchFactoryConfig | None = None,
        *,
        memory: JsonlExperimentMemory | None = None,
    ) -> None:
        self.config = config or ResearchFactoryConfig()
        self.memory = memory

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
            promotion_score=float(factory_score),
            base_score=float(result.score),
            metrics={key: float(value) for key, value in result.metrics.items()},
            robustness=float(result.robustness),
            oos_coverage=float(result.oos_coverage),
            stress_score=float(stress_score),
            stress_report=stress_report,
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

    def _apply_blind_holdout(
        self,
        bars: pd.DataFrame,
        candidates: list[FactoryCandidate],
        *,
        holdout_start: pd.Timestamp,
    ) -> tuple[list[FactoryCandidate], int]:
        passed = [candidate for candidate in candidates if candidate.gate.passed]
        selected = passed[: self.config.holdout_top_k]
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
            promotion_score = 0.70 * candidate.factory_score + 0.30 * report.score
            replacements[candidate.experiment_id] = replace(
                candidate,
                holdout_report=report,
                promotion_score=float(promotion_score),
            )

        updated = [replacements.get(candidate.experiment_id, candidate) for candidate in candidates]
        return updated, len(selected)

    def run(self, bars: pd.DataFrame, metadata: DatasetMetadata) -> FactoryReport:
        previous_best_score: float | None = None
        if self.memory is not None:
            previous = self.memory.best(only_passed=True, limit=1)
            if previous:
                previous_best_score = float(previous[0].factory_score)

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

        candidates.sort(key=lambda row: row.factory_score, reverse=True)
        holdout_evaluated = 0
        if holdout_start is not None:
            candidates, holdout_evaluated = self._apply_blind_holdout(
                bars,
                candidates,
                holdout_start=holdout_start,
            )

        if holdout_start is None:
            promotion_pool = [candidate for candidate in candidates if candidate.gate.passed]
        else:
            promotion_pool = [
                candidate
                for candidate in candidates
                if candidate.gate.passed
                and candidate.holdout_report is not None
                and candidate.holdout_report.passed
            ]
        promotion_pool.sort(key=lambda row: row.promotion_score, reverse=True)

        horizon_champions: dict[int, FactoryCandidate] = {}
        for horizon in self.config.horizons:
            winner = next((row for row in promotion_pool if row.horizon == horizon), None)
            if winner is not None:
                horizon_champions[horizon] = winner

        champion = promotion_pool[0] if promotion_pool else None
        if champion is not None and previous_best_score is not None:
            if champion.promotion_score < previous_best_score + self.config.promotion_margin:
                champion = None

        return FactoryReport(
            candidates=tuple(candidates),
            champion_candidate=champion,
            horizon_champions=horizon_champions,
            data_grade=metadata.grade,
            experiments_run=len(candidates),
            candidates_passed=len(promotion_pool),
            holdout_evaluated=holdout_evaluated,
            holdout_start=holdout_start,
        )
