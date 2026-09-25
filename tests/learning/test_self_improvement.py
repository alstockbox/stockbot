import pandas as pd
import pytest

from stockbot.learning.experiment_lifecycle import (
    ExperimentEvidence,
    ExperimentStage,
    LearningPolicy,
    advance_experiment,
)
from stockbot.learning.regret import TradeCounterfactual, analyze_regret
from stockbot.learning.stability import stability_metrics


def evidence(score=2.0, samples=200, drawdown=0.08, negative_month_rate=0.20):
    return ExperimentEvidence(
        score=score,
        metrics={
            "max_drawdown": drawdown,
            "negative_month_rate": negative_month_rate,
        },
        robustness=0.85,
        sample_count=samples,
    )


def test_backtest_advances_only_with_sufficient_evidence():
    low = advance_experiment(ExperimentStage.BACKTEST, evidence(samples=40))
    assert low.stage is ExperimentStage.BACKTEST
    assert "INSUFFICIENT_SAMPLES" in low.reasons

    good = advance_experiment(ExperimentStage.BACKTEST, evidence(samples=160))
    assert good.stage is ExperimentStage.SHADOW
    assert good.advanced


def test_higher_score_cannot_buy_worse_monthly_stability():
    champion = evidence(score=2.0, negative_month_rate=0.15)
    challenger = evidence(score=5.0, negative_month_rate=0.30)
    decision = advance_experiment(
        ExperimentStage.MICRO,
        challenger,
        champion=champion,
    )
    assert decision.stage is ExperimentStage.REJECTED
    assert "MONTH_STABILITY_WORSE_THAN_CHAMPION" in decision.reasons


def test_micro_requires_score_margin_before_promotion():
    champion = evidence(score=2.0)
    held = advance_experiment(
        ExperimentStage.MICRO,
        evidence(score=2.01),
        champion=champion,
        policy=LearningPolicy(min_score_margin=0.03),
    )
    assert held.stage is ExperimentStage.MICRO

    promoted = advance_experiment(
        ExperimentStage.MICRO,
        evidence(score=2.05),
        champion=champion,
        policy=LearningPolicy(min_score_margin=0.03),
    )
    assert promoted.stage is ExperimentStage.PROMOTED
    assert promoted.terminal


def test_calendar_month_stability_metrics():
    idx = pd.to_datetime(["2026-01-02", "2026-01-30", "2026-02-02", "2026-02-27"])
    metrics = stability_metrics(pd.Series([0.02, 0.01, -0.02, -0.01], index=idx))
    assert metrics["monthly_observations"] == 2.0
    assert metrics["negative_month_rate"] == 0.5


def test_regret_rejects_hindsight_only_counterfactual():
    with pytest.raises(ValueError):
        analyze_regret([
            TradeCounterfactual("x", -0.01, 0.02, alternative_available_at_decision=False)
        ])
