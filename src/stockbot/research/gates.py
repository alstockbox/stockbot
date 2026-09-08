from __future__ import annotations

from dataclasses import dataclass

from stockbot.arena.experiments import ModelExperimentResult
from stockbot.data.schemas import DataGrade, DatasetMetadata


@dataclass(frozen=True)
class ResearchGateCriteria:
    """Hard promotion gates for autonomous model search.

    A candidate may rank highly without being eligible for promotion. Promotion is
    intentionally conservative because repeated search across many candidates makes
    false discoveries more likely.
    """

    min_oos_coverage: float = 0.40
    min_robustness: float = 0.55
    max_drawdown: float = 0.25
    max_cvar_95: float = 0.04
    max_turnover: float = 80.0
    max_instability: float = 0.60
    max_concentration: float = 0.45
    min_factory_score: float = 0.0
    require_research_grade: bool = True


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reasons: tuple[str, ...]


def evaluate_research_gate(
    result: ModelExperimentResult,
    metadata: DatasetMetadata,
    *,
    factory_score: float,
    criteria: ResearchGateCriteria | None = None,
) -> GateDecision:
    c = criteria or ResearchGateCriteria()
    reasons: list[str] = []

    if c.require_research_grade and metadata.grade is not DataGrade.RESEARCH_GRADE:
        reasons.append("dataset_not_research_grade")
    if float(result.oos_coverage) < c.min_oos_coverage:
        reasons.append("insufficient_oos_coverage")
    if float(result.robustness) < c.min_robustness:
        reasons.append("insufficient_robustness")
    if float(result.metrics.get("max_drawdown", 1.0)) > c.max_drawdown:
        reasons.append("drawdown_limit")
    if float(result.metrics.get("cvar_95", 1.0)) > c.max_cvar_95:
        reasons.append("tail_risk_limit")
    if float(result.metrics.get("turnover", 0.0)) > c.max_turnover:
        reasons.append("turnover_limit")
    if float(result.metrics.get("instability", 0.0)) > c.max_instability:
        reasons.append("instability_limit")
    if float(result.metrics.get("concentration", 0.0)) > c.max_concentration:
        reasons.append("concentration_limit")
    if float(factory_score) <= c.min_factory_score:
        reasons.append("non_positive_factory_score")

    return GateDecision(passed=not reasons, reasons=tuple(reasons))
