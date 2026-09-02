import pandas as pd

from stockbot.strategies.baselines import BreakoutStrategy, MeanReversionStrategy, MomentumStrategy, TrendStrategy


def test_trend_and_momentum_reward_positive_direction():
    row = pd.Series({"momentum_20": 0.10, "sma_20_dist": 0.06, "realized_vol_20": 0.20})
    assert TrendStrategy().generate_signal(row) > 0.5
    assert MomentumStrategy().generate_signal(row) > 0.5


def test_mean_reversion_prefers_short_term_selloff_in_long_only_mode():
    oversold = pd.Series({"momentum_5": -0.08, "realized_vol_20": 0.20})
    overbought = pd.Series({"momentum_5": 0.08, "realized_vol_20": 0.20})
    assert MeanReversionStrategy().generate_signal(oversold) > MeanReversionStrategy().generate_signal(overbought)


def test_breakout_requires_positive_prior_range_break():
    assert BreakoutStrategy().generate_signal(pd.Series({"breakout_20": 0.03, "realized_vol_20": 0.2})) > 0.5
    assert BreakoutStrategy().generate_signal(pd.Series({"breakout_20": -0.03, "realized_vol_20": 0.2})) < 0.5
