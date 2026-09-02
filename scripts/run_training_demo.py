from __future__ import annotations

import numpy as np
import pandas as pd

from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.research.training_pipeline import run_training_research


def make_demo_bars(seed: int = 20260903) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=420, freq="B", tz="UTC")
    symbols = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA", "ETA", "THETA", "IOTA", "KAPPA"]
    rows: list[dict[str, object]] = []
    market = rng.normal(0.00025, 0.0075, len(dates))
    for j, symbol in enumerate(symbols):
        cyclical = 0.0012 * np.sin(np.arange(len(dates)) / (11.0 + j))
        idiosyncratic = rng.normal(0, 0.006 + j * 0.00015, len(dates))
        quality_drift = 0.00003 * j
        log_returns = market + cyclical + idiosyncratic + quality_drift
        close = 100.0 * np.exp(np.cumsum(log_returns))
        volumes = rng.integers(600_000, 5_000_000, len(dates))
        for i, dt in enumerate(dates):
            rows.append({"symbol": symbol,"timestamp": dt,"open": float(close[i] * (1 + rng.normal(0, 0.001))),"high": float(close[i] * 1.008),"low": float(close[i] * 0.992),"close": float(close[i]),"volume": int(volumes[i])})
    return pd.DataFrame(rows)


def main() -> None:
    metadata = DatasetMetadata(name="stockbot-v1-synthetic-demo",source="scripts/run_training_demo.py",grade=DataGrade.DEMO,version="1")
    run = run_training_research(make_demo_bars(), metadata, horizon=5)
    print("StockBot V1 Training Demo")
    print("DATA GRADE: DEMO / NON-RESEARCH-GRADE")
    print(f"dataset fingerprint: {run.dataset_fingerprint[:16]}...")
    print(f"horizon: {run.horizon} trading days")
    print("\nOOS Alpha Arena")
    for rank, row in enumerate(run.leaderboard, 1):
        print(f"{rank:>2}. {row.name:<14} score={row.score:>8.3f} OOS={row.oos_coverage:>6.1%} CAGR={row.metrics['cagr']:>7.2%} Sharpe={row.metrics['sharpe']:>6.2f} MaxDD={row.metrics['max_drawdown']:>7.2%} turnover={row.metrics['turnover']:>7.1f}")
    print("\nPromotion candidate:", run.champion_candidate.name if run.champion_candidate else "NONE")
    print("Demo data is intentionally ineligible for champion promotion.")


if __name__ == "__main__":
    main()
