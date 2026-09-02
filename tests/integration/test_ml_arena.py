import numpy as np
import pandas as pd

from stockbot.research.pipeline import run_research


def test_research_arena_includes_walk_forward_ml_challenger_and_champion():
    rng = np.random.default_rng(17)
    n = 420
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    latent = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(0.0004 + latent))
    frame = pd.DataFrame({
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.integers(1_000_000, 4_000_000, n),
    }, index=idx)
    run = run_research(frame)
    names = {row["name"] for row in run.leaderboard}
    assert "ml_ridge" in names
    assert run.ml_oos_samples >= 50
    assert run.champion_name in names
