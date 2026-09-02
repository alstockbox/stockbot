from stockbot.ensemble.weighted import combine_signals
from stockbot.portfolio.constructor import PortfolioConfig, target_from_signal


def test_weighted_ensemble_combines_signal_strength():
    assert combine_signals({"trend": 1.0, "meanrev": 0.0}, {"trend": 0.8, "meanrev": 0.2}) == 0.8


def test_ensemble_can_hold_cash_when_strategy_budget_is_below_one():
    assert combine_signals({"trend": 1.0}, {"trend": 0.3}) == 0.3


def test_portfolio_sizing_respects_hard_position_cap():
    cfg = PortfolioConfig(target_volatility=0.20, max_position_weight=0.50)
    target = target_from_signal("SPY", signal=0.9, volatility=0.10, config=cfg)
    assert target.weight == 0.50
