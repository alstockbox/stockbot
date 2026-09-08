import numpy as np
import pandas as pd

from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.ml.models import ModelConfig
from stockbot.research.feature_ablation import evaluate_feature_group_ablation


def _bars() -> pd.DataFrame:
    rng = np.random.default_rng(77)
    dates = pd.date_range("2024-01-02", periods=240, freq="B", tz="UTC")
    rows = []
    market = rng.normal(0.0002, 0.006, len(dates))
    for j, symbol in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]):
        noise = rng.normal(0.0, 0.0045 + j * 0.0002, len(dates))
        close = 100.0 * np.exp(np.cumsum(market + noise + 0.00002 * j))
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


def test_feature_ablation_refits_same_model_without_selected_groups():
    metadata = DatasetMetadata(
        name="ablation-test",
        source="synthetic",
        grade=DataGrade.RESEARCH_GRADE,
    )
    groups = {
        "volatility": ("realized_vol_5", "realized_vol_20", "realized_vol_60"),
        "microstructure": ("range_1", "range_20", "gap_1"),
    }
    report = evaluate_feature_group_ablation(
        _bars(),
        metadata,
        ModelConfig("ridge", {"alpha": 1.0}, seed=7),
        horizon=5,
        groups=groups,
        train_periods=84,
        test_periods=20,
    )

    assert len(report.results) == 2
    assert {row.group for row in report.results} == {"volatility", "microstructure"}
    assert all(np.isfinite(row.ablated_score) for row in report.results)
    assert all(np.isfinite(row.score_impact) for row in report.results)
    assert set(report.recommended_drop_groups).issubset(groups)
