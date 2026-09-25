from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from stockbot.ai.researcher import ResearchScientist
from stockbot.learning.regret import RegretSummary, TradeCounterfactual, analyze_regret
from stockbot.learning.stability import stability_metrics
from stockbot.llm.schemas import ResearchHypothesis


@dataclass(frozen=True)
class DailyLearningReport:
    metrics: dict[str, float]
    regret: RegretSummary
    hypotheses: tuple[ResearchHypothesis, ...]
    regime_degradation: bool


def build_daily_learning_report(
    returns: pd.Series,
    counterfactuals: list[TradeCounterfactual],
    *,
    base_metrics: dict[str, float] | None = None,
    regime_degradation: bool = False,
    scientist: ResearchScientist | None = None,
) -> DailyLearningReport:
    metrics = dict(base_metrics or {})
    metrics.update(stability_metrics(returns))
    regret = analyze_regret(counterfactuals)
    reviewer = scientist or ResearchScientist()
    hypotheses = reviewer.review(
        {
            "metrics": metrics,
            "regret": asdict(regret),
            "regime_degradation": regime_degradation,
        }
    )
    return DailyLearningReport(
        metrics=metrics,
        regret=regret,
        hypotheses=tuple(hypotheses),
        regime_degradation=bool(regime_degradation),
    )
