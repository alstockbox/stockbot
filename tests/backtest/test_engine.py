import pandas as pd

from stockbot.backtest.engine import BacktestConfig, run_backtest


def test_backtest_uses_previous_bar_signal_to_prevent_same_bar_lookahead():
    idx = pd.date_range("2025-01-01", periods=3, freq="B")
    frame = pd.DataFrame({"close": [100.0, 110.0, 110.0]}, index=idx)
    signals = pd.Series([0.0, 1.0, 0.0], index=idx)
    result = run_backtest(frame, signals, BacktestConfig(commission_bps=0, slippage_bps=0))
    assert result.equity_curve.iloc[-1] == 10_000.0


def test_costs_reduce_equity_when_turnover_occurs():
    idx = pd.date_range("2025-01-01", periods=4, freq="B")
    frame = pd.DataFrame({"close": [100.0, 100.0, 100.0, 100.0]}, index=idx)
    signals = pd.Series([1.0, 0.0, 1.0, 0.0], index=idx)
    free = run_backtest(frame, signals, BacktestConfig(commission_bps=0, slippage_bps=0))
    costly = run_backtest(frame, signals, BacktestConfig(commission_bps=5, slippage_bps=5))
    assert costly.equity_curve.iloc[-1] < free.equity_curve.iloc[-1]
    assert costly.total_cost > 0


def test_backtest_can_enforce_hard_drawdown_risk_gate():
    from stockbot.risk.engine import RiskConfig, RiskEngine

    idx = pd.date_range("2025-01-01", periods=5, freq="B")
    frame = pd.DataFrame({"close": [100.0, 90.0, 81.0, 81.0, 81.0]}, index=idx)
    signals = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=idx)
    cfg = BacktestConfig(commission_bps=0, slippage_bps=0)
    unguarded = run_backtest(frame, signals, cfg)
    guarded = run_backtest(
        frame,
        signals,
        cfg,
        risk_engine=RiskEngine(RiskConfig(max_drawdown=0.095)),
    )
    assert guarded.final_equity > unguarded.final_equity
    assert guarded.positions.iloc[2] == 0.0
