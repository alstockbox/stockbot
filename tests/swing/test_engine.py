from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockbot.swing.engine import (
    SwingConfig, run_swing_backtest, validate_ohlcv,
)


def sample_bars(n: int = 240, *, symbols: tuple[str, ...] = ("AAA",)) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=n)
    rows = []
    for symbol in symbols:
        for i, date in enumerate(dates):
            close = 20.0 + i * (0.10 if symbol == "AAA" else 0.11)
            rows.append({
                "date": date, "symbol": symbol,
                "open": close - 0.05, "high": close + 0.10,
                "low": close - 0.10, "close": close, "volume": 10000,
            })
    return pd.DataFrame(rows)


def test_inputs_reject_duplicate_or_invalid_ohlc():
    bars = sample_bars()
    with pytest.raises(ValueError, match="duplicate"):
        validate_ohlcv(pd.concat([bars, bars.iloc[[0]]], ignore_index=True))
    bad = bars.copy()
    bad.loc[0, "high"] = bad.loc[0, "close"] - 1
    with pytest.raises(ValueError, match="OHLC"):
        validate_ohlcv(bad)


def test_inputs_fail_closed_on_missing_symbol_session():
    bars = sample_bars(symbols=("AAA", "BBB"))
    bars = bars.drop(bars[(bars["symbol"] == "BBB") & (bars["date"] == bars["date"].iloc[5])].index)
    with pytest.raises(ValueError, match="identical session"):
        validate_ohlcv(bars)


def test_next_open_execution_whole_shares_and_fee_accounting():
    bars = sample_bars()
    cfg = SwingConfig(initial_sek=1000, max_position_weight=0.2,
                      execution_bps_each_side=0, commission_rate=0.0025,
                      minimum_commission_sek=1)
    result = run_swing_backtest(bars, cfg)
    buys = [f for f in result.fills if f.side == "BUY"]
    assert buys, "synthetic continuous trend should generate entries"
    assert buys[0].date > "2025-01-01"
    assert buys[0].quantity >= 1 and isinstance(buys[0].quantity, int)
    first_date = pd.Timestamp(buys[0].date)
    first_open = bars.loc[bars["date"] == first_date, "open"].iloc[0]
    assert buys[0].executed_price == pytest.approx(first_open)
    assert result.total_fees_sek == pytest.approx(sum(f.fee_sek for f in result.fills))
    assert np.isfinite(result.equity.to_numpy()).all()
    assert result.equity.min() > 0
    assert 0 <= result.max_drawdown < 1


def test_costs_reduce_net_result_in_rising_market():
    bars = sample_bars()
    no_cost = run_swing_backtest(bars, SwingConfig(commission_rate=0,
                            minimum_commission_sek=0, execution_bps_each_side=0))
    cost = run_swing_backtest(bars, SwingConfig(commission_rate=0.0025,
                            minimum_commission_sek=1, execution_bps_each_side=10))
    assert cost.final_equity_sek < no_cost.final_equity_sek
    assert cost.total_fees_sek > 0
    assert cost.total_execution_impact_sek > 0


def test_flat_market_does_not_trade():
    bars = sample_bars()
    for name in ("open", "high", "low", "close"):
        bars[name] = 30.0
    result = run_swing_backtest(bars)
    assert result.fills == ()
    assert result.final_equity_sek == 1_000.0


def test_configuration_is_paper_conservative():
    with pytest.raises(ValueError):
        SwingConfig(initial_sek=-10)
    with pytest.raises(ValueError):
        SwingConfig(risk_per_trade=0.05)
    with pytest.raises(ValueError):
        SwingConfig(max_position_weight=1.1)


def test_prior_close_relative_strength_determines_first_position():
    bars = sample_bars(symbols=("AAA", "BBB"))
    cfg = SwingConfig(max_positions=1)
    result = run_swing_backtest(bars, cfg)
    first = next(f for f in result.fills if f.side == "BUY")
    assert first.symbol == "BBB", "greater prior-close strength must win, not ticker order"
