import numpy as np
import pandas as pd

from stockbot.research.pipeline import run_research


def test_end_to_end_research_pipeline_returns_ranked_models_and_hypotheses():
    rng = np.random.default_rng(11)
    n = 320
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    log_returns = 0.0005 + 0.002 * np.sin(np.arange(n) / 18) + rng.normal(0, 0.009, n)
    close = 100 * np.exp(np.cumsum(log_returns))
    frame = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.integers(1_000_000, 3_000_000, n),
    }, index=idx)
    run = run_research(frame)
    assert len(run.leaderboard) >= 4
    assert np.isfinite(run.ensemble.metrics["cagr"])
    assert 0 <= run.latest_signal <= 1
    assert run.hypotheses


def test_research_pipeline_routes_ensemble_through_hard_risk_gate():
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.concatenate([
        np.linspace(100.0, 140.0, 80),
        np.array([30.0]),
        np.full(n - 81, 30.0),
    ])
    frame = pd.DataFrame({
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.full(n, 2_000_000),
    }, index=idx)
    run = run_research(frame)
    assert (run.ensemble.positions.iloc[82:] == 0.0).all()
