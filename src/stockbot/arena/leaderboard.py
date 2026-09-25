from __future__ import annotations

from collections.abc import Iterable
import math

from stockbot.arena.experiments import ModelExperimentResult
from stockbot.data.schemas import DataGrade, DatasetMetadata


def rank_experiments(results: Iterable[ModelExperimentResult]) -> list[ModelExperimentResult]:
    return sorted(list(results), key=lambda result: float(result.score), reverse=True)


def _metric(result: ModelExperimentResult, key: str) -> float | None:
    if key not in result.metrics:
        return None
    value = float(result.metrics[key])
    return value if math.isfinite(value) else None


def eligible_for_promotion(result: ModelExperimentResult, metadata: DatasetMetadata) -> bool:
    if metadata.grade is not DataGrade.RESEARCH_GRADE:
        return False
    if not math.isfinite(float(result.score)) or float(result.score) <= 0.0:
        return False
    if result.oos_coverage < 0.30:
        return False
    if result.robustness < 0.50:
        return False

    drawdown = _metric(result, "max_drawdown")
    turnover = _metric(result, "turnover")
    negative_month_rate = _metric(result, "negative_month_rate")
    monthly_observations = _metric(result, "monthly_observations")
    if drawdown is None or drawdown > 0.30:
        return False
    if turnover is None or turnover > 100.0:
        return False
    if negative_month_rate is None or negative_month_rate > 0.50:
        return False
    if monthly_observations is None or monthly_observations < 6:
        return False
    return True
