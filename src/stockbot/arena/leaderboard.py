from __future__ import annotations

from collections.abc import Iterable
from stockbot.arena.experiments import ModelExperimentResult
from stockbot.data.schemas import DataGrade, DatasetMetadata


def rank_experiments(results: Iterable[ModelExperimentResult]) -> list[ModelExperimentResult]:
    return sorted(list(results), key=lambda result: float(result.score), reverse=True)


def eligible_for_promotion(result: ModelExperimentResult, metadata: DatasetMetadata) -> bool:
    if metadata.grade is not DataGrade.RESEARCH_GRADE: return False
    if result.oos_coverage < 0.30: return False
    if result.robustness < 0.50: return False
    if float(result.metrics.get("max_drawdown", 1.0)) > 0.30: return False
    if float(result.metrics.get("turnover", 0.0)) > 100.0: return False
    return float(result.score) > 0.0
