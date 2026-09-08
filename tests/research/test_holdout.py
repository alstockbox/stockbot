import numpy as np
import pandas as pd

from stockbot.ml.models import ModelConfig
from stockbot.research.holdout import HoldoutConfig, evaluate_blind_holdout, split_research_holdout


def _bars() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2023-01-02", periods=260, freq="B", tz="UTC")
    rows = []
    for j, symbol in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]):
        noise = rng.normal(0.0, 0.008, len(dates))
        drift = 0.00015 + j * 0.00004
        close = 100.0 * np.exp(np.cumsum(drift + noise))
        volume = rng.integers(600_000, 2_000_000, len(dates))
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": close[i],
                    "high": close[i] * 1.01,
                    "low": close[i] * 0.99,
                    "close": close[i],
                    "volume": volume[i],
                }
            )
    return pd.DataFrame(rows)


def test_split_reserves_final_time_block_before_research():
    bars = _bars()
    cfg = HoldoutConfig(fraction=0.15, min_holdout_periods=30, min_research_periods=160)
    research, holdout_start = split_research_holdout(bars, cfg)

    research_dates = pd.to_datetime(research["timestamp"], utc=True)
    all_dates = pd.DatetimeIndex(pd.to_datetime(bars["timestamp"], utc=True).unique()).sort_values()

    assert research_dates.max() < holdout_start
    assert (all_dates >= holdout_start).sum() >= cfg.min_holdout_periods


def test_blind_holdout_scores_only_reserved_final_block():
    bars = _bars()
    cfg = HoldoutConfig(
        fraction=0.15,
        min_holdout_periods=30,
        min_research_periods=160,
        min_prediction_coverage=0.40,
        max_drawdown=0.80,
        min_stress_score=0.0,
        min_score=-100.0,
    )
    _, holdout_start = split_research_holdout(bars, cfg)
    report = evaluate_blind_holdout(
        bars,
        ModelConfig("ridge", {"alpha": 1.0}, seed=7),
        horizon=5,
        holdout_start=holdout_start,
        config=cfg,
    )

    assert report.holdout_start == holdout_start
    assert report.periods >= cfg.min_holdout_periods
    assert report.prediction_coverage >= cfg.min_prediction_coverage
    assert 0.0 <= report.stress_score <= 1.0
    assert np.isfinite(report.score)
    assert report.passed
