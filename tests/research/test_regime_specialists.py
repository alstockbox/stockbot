import numpy as np
import pandas as pd

from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.domain.models import MarketRegime
from stockbot.ml.models import ModelConfig
from stockbot.research.regime_specialists import run_regime_specialist


def _bars() -> pd.DataFrame:
    rng = np.random.default_rng(1234)
    dates = pd.date_range("2023-01-02", periods=300, freq="B", tz="UTC")
    rows = []
    market = rng.normal(0.0002, 0.006, len(dates))
    for j, symbol in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]):
        noise = rng.normal(0.0, 0.004 + j * 0.0002, len(dates))
        close = 100.0 * np.exp(np.cumsum(market + noise + 0.00002 * j))
        volume = rng.integers(700_000, 2_500_000, len(dates))
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": close[i],
                    "high": close[i] * 1.008,
                    "low": close[i] * 0.992,
                    "close": close[i],
                    "volume": volume[i],
                }
            )
    return pd.DataFrame(rows)


def test_neutral_regime_specialist_runs_with_purged_oos_evaluation():
    metadata = DatasetMetadata(
        name="specialist-test",
        source="synthetic",
        grade=DataGrade.RESEARCH_GRADE,
    )
    result = run_regime_specialist(
        _bars(),
        metadata,
        ModelConfig("ridge", {"alpha": 1.0}, seed=7),
        regime=MarketRegime.NEUTRAL_CHOP,
        horizon=5,
    )

    assert result.regime is MarketRegime.NEUTRAL_CHOP
    assert result.horizon == 5
    assert np.isfinite(result.score)
    assert 0.0 <= result.oos_coverage <= 1.0
    assert len(result.net_returns) > 0
    assert 0.0 <= result.stress_report.score <= 1.0
