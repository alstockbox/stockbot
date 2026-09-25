import pandas as pd

from stockbot.learning.daily_engine import build_daily_learning_report
from stockbot.learning.regret import TradeCounterfactual


def test_daily_report_generates_regret_hypothesis():
    idx = pd.date_range("2026-01-01", periods=40, freq="D")
    rows = [
        TradeCounterfactual("a", -0.01, 0.00, True),
        TradeCounterfactual("b", 0.00, 0.01, True),
        TradeCounterfactual("c", 0.002, 0.004, True),
    ]
    report = build_daily_learning_report(
        pd.Series([0.001] * 20 + [-0.001] * 20, index=idx),
        rows,
    )
    assert report.regret.total_regret > 0.0
    assert any(item.category == "decision_regret" for item in report.hypotheses)
