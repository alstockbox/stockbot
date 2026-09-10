import numpy as np
import pandas as pd

from stockbot.research.ensemble import build_horizon_ensemble


def test_ensemble_downweights_redundant_highly_correlated_members():
    index = pd.date_range("2024-01-01", periods=252, freq="B", tz="UTC")
    x = np.linspace(0.0, 12.0 * np.pi, len(index))
    a = pd.Series(0.0005 + 0.004 * np.sin(x), index=index)
    b = pd.Series(0.0005 + 0.0041 * np.sin(x), index=index)
    c = pd.Series(0.0005 + 0.004 * np.cos(x), index=index)

    report = build_horizon_ensemble(
        {"a": a, "b": b, "c": c},
        {"a": 1.0, "b": 1.0, "c": 1.0},
    )

    assert abs(sum(report.member_weights.values()) - 1.0) < 1e-9
    assert report.member_weights["c"] > report.member_weights["a"]
    assert report.member_weights["c"] > report.member_weights["b"]
