import pandas as pd

from stockbot.research.failures import mine_hard_negatives
from stockbot.research.stress import evaluate_stress_suite


def test_hard_negative_miner_prioritizes_confident_wrong_calls():
    idx = pd.RangeIndex(6)
    predictions = pd.Series([0.01, 0.02, 0.08, -0.09, 0.005, -0.03], index=idx)
    labels = pd.Series([0.02, 0.01, -0.04, 0.05, -0.002, -0.01], index=idx)

    hard = mine_hard_negatives(predictions, labels, confidence_quantile=0.50)

    assert not hard.empty
    assert set(hard.index).issubset({2, 3})
    assert (hard["severity"] > 0).all()


def test_stress_suite_produces_bounded_robustness_score():
    index = pd.date_range("2025-01-01", periods=252, freq="B")
    returns = pd.Series(0.0008, index=index)
    returns.iloc[::31] = -0.012
    turnover = pd.Series(0.10, index=index)

    report = evaluate_stress_suite(returns, turnover=turnover)

    assert 0.0 <= report.score <= 1.0
    assert 0.0 <= report.survival_rate <= 1.0
    assert report.worst_drawdown >= 0.0
    assert "combined_adverse" in report.scenario_metrics
