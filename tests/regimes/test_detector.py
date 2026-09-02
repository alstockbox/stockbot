import pandas as pd

from stockbot.domain.models import MarketRegime
from stockbot.regimes.detector import detect_regime


def test_detects_bull_trend():
    row = pd.Series({"momentum_20": 0.08, "sma_20_dist": 0.05, "realized_vol_20": 0.18})
    assert detect_regime(row) is MarketRegime.BULL_TREND


def test_detects_bear_stress():
    row = pd.Series({"momentum_20": -0.08, "sma_20_dist": -0.05, "realized_vol_20": 0.70})
    assert detect_regime(row) is MarketRegime.BEAR_STRESS


def test_defaults_to_neutral_chop():
    row = pd.Series({"momentum_20": 0.001, "sma_20_dist": 0.001, "realized_vol_20": 0.20})
    assert detect_regime(row) is MarketRegime.NEUTRAL_CHOP
