from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class ExperimentStage(str, Enum):
    PROPOSED = "proposed"
    BACKTEST = "backtest"
    SHADOW = "shadow"
    MICRO = "micro"
    PROMOTED = "promoted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ExperimentEvidence:
    score: float
    metrics: dict[str, float]
    robustness: float
    sample_count: int


@dataclass(frozen=True)
class LearningPolicy:
    min_backtest_samples: int = 126
    min_shadow_samples: int = 20
    min_micro_samples: int = 20
    min_robustness: float = 0.60
    min_score_margin: float = 0.03
    max_drawdown: float = 0.20
    max_negative_month_rate: float = 0.50
    max_drawdown_deterioration: float = 0.02
    max_negative_month_deterioration: float = 0.02


@dataclass(frozen=True)
class StageDecision:
    stage: ExperimentStage
    advanced: bool
    terminal: bool
    reasons: tuple[str, ...] = ()


def _metric(evidence: ExperimentEvidence, key: str) -> float:
    value = float(evidence.metrics.get(key, 0.0))
    return value if math.isfinite(value) else float("inf")


def _quality_failures(
    evidence: ExperimentEvidence,
    policy: LearningPolicy,
    champion: ExperimentEvidence | None,
) -> list[str]:
    failures: list[str] = []
    if not math.isfinite(evidence.score):
        failures.append("NON_FINITE_SCORE")
    if evidence.robustness < policy.min_robustness:
        failures.append("LOW_ROBUSTNESS")

    drawdown = _metric(evidence, "max_drawdown")
    negative_month_rate = _metric(evidence, "negative_month_rate")
    if drawdown > policy.max_drawdown:
        failures.append("MAX_DRAWDOWN")
    if negative_month_rate > policy.max_negative_month_rate:
        failures.append("NEGATIVE_MONTH_RATE")

    if champion is not None:
        champion_dd = _metric(champion, "max_drawdown")
        champion_loss_rate = _metric(champion, "negative_month_rate")
        if drawdown > champion_dd + policy.max_drawdown_deterioration:
            failures.append("DRAWDOWN_WORSE_THAN_CHAMPION")
        if negative_month_rate > champion_loss_rate + policy.max_negative_month_deterioration:
            failures.append("MONTH_STABILITY_WORSE_THAN_CHAMPION")
    return failures


def advance_experiment(
    stage: ExperimentStage,
    evidence: ExperimentEvidence | None = None,
    *,
    champion: ExperimentEvidence | None = None,
    policy: LearningPolicy | None = None,
) -> StageDecision:
    policy = policy or LearningPolicy()
    if stage in {ExperimentStage.PROMOTED, ExperimentStage.REJECTED}:
        return StageDecision(stage, False, True, ("TERMINAL_STAGE",))
    if stage is ExperimentStage.PROPOSED:
        return StageDecision(ExperimentStage.BACKTEST, True, False)
    if evidence is None:
        return StageDecision(stage, False, False, ("MISSING_EVIDENCE",))

    failures = _quality_failures(evidence, policy, champion)
    if failures:
        return StageDecision(ExperimentStage.REJECTED, False, True, tuple(failures))

    required = {
        ExperimentStage.BACKTEST: policy.min_backtest_samples,
        ExperimentStage.SHADOW: policy.min_shadow_samples,
        ExperimentStage.MICRO: policy.min_micro_samples,
    }[stage]
    if evidence.sample_count < required:
        return StageDecision(stage, False, False, ("INSUFFICIENT_SAMPLES",))

    if stage is ExperimentStage.MICRO and champion is not None:
        if evidence.score < champion.score + policy.min_score_margin:
            return StageDecision(stage, False, False, ("INSUFFICIENT_SCORE_MARGIN",))

    next_stage = {
        ExperimentStage.BACKTEST: ExperimentStage.SHADOW,
        ExperimentStage.SHADOW: ExperimentStage.MICRO,
        ExperimentStage.MICRO: ExperimentStage.PROMOTED,
    }[stage]
    return StageDecision(next_stage, True, next_stage is ExperimentStage.PROMOTED)
