import numpy as np
import pandas as pd

from stockbot.evaluation.metrics import performance_metrics


def test_metrics_return_finite_risk_adjusted_statistics():
    returns = pd.Series([0.01, -0.02, 0.015, 0.005, -0.004] * 60)
    benchmark = pd.Series([0.004, -0.01, 0.008, 0.002, -0.002] * 60)
    metrics = performance_metrics(returns, benchmark)
    for key in ["cagr", "volatility", "sharpe", "sortino", "max_drawdown", "calmar", "excess_return", "cvar_95"]:
        assert key in metrics
        assert np.isfinite(metrics[key])
    assert metrics["max_drawdown"] > 0
