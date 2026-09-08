from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from stockbot.arena.experiments import ModelExperimentResult
from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.ml.models import ModelConfig
from stockbot.research.gates import GateDecision, ResearchGateCriteria, evaluate_research_gate
from stockbot.research.memory import JsonlExperimentMemory, experiment_id, make_record
from stockbot.research.objective import ObjectiveWeights, risk_adjusted_objective
from stockbot.research.population import ModelPopulationConfig, generate_model_population
from stockbot.research.training_pipeline import run_training_research


@dataclass(frozen=True)
class ResearchFactoryConfig:
    """Configuration for large-scale autonomous challenger search."""

    horizons: tuple[int, ...] = (1, 5, 20)
    population: ModelPopulationConfig = field(default_factory=ModelPopulationConfig)
    gates: ResearchGateCriteria = field(default_factory=ResearchGateCriteria)
    objective: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    promotion_margin: float = 0.02

    def __post_init__(self) -> None:
        if not self.horizons or any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")
        if self.promotion_margin < 0:
            raise ValueError("promotion_margin must be non-negative")


@dataclass(frozen=True)
class FactoryCandidate:
    experiment_id: str
    horizon: int
    model_name: str
    model_params: dict[str, Any]
    seed: int
    factory_score: float
    base_score: float
    metrics: dict[str, float]
    robustness: float
    oos_coverage: float
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


class ResearchFactory:
    """Run broad, multi-horizon ML research and promote only robust OOS candidates.

    The factory expands the existing V1 training pipeline rather than replacing it.
    Every challenger still uses causal cross-sectional features and purged walk-forward
    OOS evaluation. The factory adds population search, a stronger objective, hard
    gates, multi-horizon competition and persistent research memory.
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

    def _candidate(
        self,
        result: ModelExperimentResult,
        metadata: DatasetMetadata,
        *,
        horizon: int,
    ) -> FactoryCandidate:
        model = self._model_from_result(result)
        factory_score = risk_adjusted_objective(
            result.metrics,
            robustness=result.robustness,
            oos_coverage=result.oos_coverage,
            weights=self.config.objective,
        )
        gate = evaluate_research_gate(
            result,
            metadata,
            factory_score=factory_score,
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
            base_score=float(result.score),
            metrics={key: float(value) for key, value in result.metrics.items()},
            robustness=float(result.robustness),
            oos_coverage=float(result.oos_coverage),
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
                )
            )
        return candidate

    def run(self, bars: pd.DataFrame, metadata: DatasetMetadata) -> FactoryReport:
        population = generate_model_population(self.config.population)
        candidates: list[FactoryCandidate] = []

        for horizon in self.config.horizons:
            training_run = run_training_research(
                bars,
                metadata,
                model_configs=population,
                horizon=horizon,
            )
            candidates.extend(
                self._candidate(result, metadata, horizon=horizon)
                for result in training_run.leaderboard
            )

        candidates.sort(key=lambda row: row.factory_score, reverse=True)
        passed = [row for row in candidates if row.gate.passed]

        horizon_champions: dict[int, FactoryCandidate] = {}
        for horizon in self.config.horizons:
            winner = next((row for row in passed if row.horizon == horizon), None)
            if winner is not None:
                horizon_champions[horizon] = winner

        champion = passed[0] if passed else None
        if champion is not None and self.memory is not None:
            incumbents = [
                row
                for row in self.memory.best(only_passed=True, limit=100)
                if row.experiment_id != champion.experiment_id
            ]
            if incumbents:
                incumbent_score = max(row.factory_score for row in incumbents)
                if champion.factory_score < incumbent_score + self.config.promotion_margin:
                    champion = None

        return FactoryReport(
            candidates=tuple(candidates),
            champion_candidate=champion,
            horizon_champions=horizon_champions,
            data_grade=metadata.grade,
            experiments_run=len(candidates),
            candidates_passed=len(passed),
        )
