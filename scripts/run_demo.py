from __future__ import annotations

import json

import numpy as np
import pandas as pd

from stockbot.research.pipeline import run_research


def synthetic_market(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    cycle = 0.0018 * np.sin(np.arange(n) / 22.0)
    log_returns = 0.00045 + cycle + rng.normal(0.0, 0.009, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    return pd.DataFrame(
        {
            "open": close * (1.0 + rng.normal(0.0, 0.001, n)),
            "high": close * (1.0 + rng.uniform(0.003, 0.015, n)),
            "low": close * (1.0 - rng.uniform(0.003, 0.015, n)),
            "close": close,
            "volume": rng.integers(1_000_000, 6_000_000, n),
        },
        index=idx,
    )


def main() -> None:
    run = run_research(synthetic_market())
    payload = {
        "latest_regime": run.latest_regime.value,
        "latest_signal": round(run.latest_signal, 4),
        "champion_name": run.champion_name,
        "ml_oos_samples": run.ml_oos_samples,
        "ensemble_metrics": {k: round(v, 6) for k, v in run.ensemble.metrics.items()},
        "leaderboard": [
            {"name": row["name"], "score": round(float(row["score"]), 4)}
            for row in run.leaderboard
        ],
        "research_hypotheses": [idea.model_dump() for idea in run.hypotheses],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
