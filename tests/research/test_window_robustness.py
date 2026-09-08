import numpy as np
import pandas as pd

from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.ml.models import ModelConfig
from stockbot.research.window_robustness import evaluate_training_window_robustness


def _bars() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2024-01-02", periods=280, freq="B", tz="UTC")
    rows = []
    market = rng.normal(0.0002, 0.006, len(dates))
    for j, symbol in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]):
        noise = rng.normal(0.0, 0.0045 + j * 0.0002, len(dates))
        close = 100.0 * np.exp(np.cumsum(market + noise + 0.00003 * j))
        volume = rng.integers(600_000, 2_000_000, len(dates))
        for i, dt in enumerate(dates):
            rows.append({
                "symbol": symbol,
                "timestamp": dt,
                "open": close[i],
                "high": close[i] * 1.008,
                "low": close[i] * 0.992,
                "close": close[i],
                "volume": volume[i],
            })
    return pd.DataFrame(rows)


def test_training_window_robustness_reuses_same_model_across_memory_lengths():
    metadata = DatasetMetadata(
        name="window-test",
        source="synthetic",
        grade=DataGrade.RESEARCH_GRADE,
    )
    report = evaluate_training_window_robustness(
        _bars(),
        metadata,
        ModelConfig("ridge", {"alpha": 1.0}, seed=7),
        horizon=5,
        train_windows=(84, 126, 200),
        test_periods=20,
    )

    assert len(report.results) == 3
    assert {row.train_periods for row in report.results} == {84, 126, 200}
    assert all(np.isfinite(row.score) for row in report.results)
    assert 0.0 <= report.score <= 1.0
    assert 0.0 <= report.positive_fraction <= 1.0
