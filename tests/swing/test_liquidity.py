"""Risk and liquidity regression tests; all data is synthetic and non-executable."""
from __future__ import annotations

import pandas as pd
import pytest

from stockbot.swing.engine import SwingConfig, run_swing_backtest


def bars(n: int = 260, *, volume: float = 10_000) -> pd.DataFrame:
    days = pd.bdate_range("2025-01-02", periods=n)
    rows = []
    for i, date in enumerate(days):
        p = 20 + i * .11
        rows.append({
            "date": date, "symbol": "TEST",
            "open": p - .04, "high": p + .05,
            "low": p - .12, "close": p,
            "volume": volume,
        })
    return pd.DataFrame(rows)


def test_partial_stop_exit_carries_remaining_shares_to_next_session():
    df = bars()
    days = df["date"]
    # A crash below the known trailing stop, with only 100 total shares
    # traded per day. Under 2% participation cap, at most 2 may be sold.
    for i in range(220, 224):
        df.loc[i, ["open", "close", "high", "low"]] = [
            30.0, 30.0, 30.2, 29.8
        ]
        df.loc[i, "volume"] = 100
    result = run_swing_backtest(df, SwingConfig(
        max_daily_volume_fraction=.02,
        min_prior_median_turnover_sek=100_000,
        execution_bps_each_side=0,
    ))
    buys = [f for f in result.fills if f.side == "BUY"]
    sells = [f for f in result.fills if f.side == "SELL"]
    assert buys and buys[0].date < str(days[220].date())
    crash_sales = [f for f in sells if f.date in {
        str(days[220].date()), str(days[221].date()), str(days[222].date())
    }]
    assert crash_sales
    assert all(fill.quantity <= 2 for fill in crash_sales)
    assert result.partial_exit_events > 0
    assert any(f.reason == "EXIT_RETRY" for f in sells)
    assert result.total_fees_sek == pytest.approx(sum(f.fee_sek for f in result.fills))


def test_no_liquidity_means_no_entry_not_fictitious_fill():
    low = bars(volume=49)
    result = run_swing_backtest(low, SwingConfig(
        min_prior_median_turnover_sek=0,
        max_daily_volume_fraction=.02,
    ))
    assert result.fills == ()
    assert result.liquidity_rejected_entries > 0
    assert result.final_equity_sek == 1000.0


def test_prior_turnover_filter_does_not_see_future_volume():
    df = bars(volume=100)
    restricted = run_swing_backtest(df, SwingConfig(
        min_prior_median_turnover_sek=100_000,
        max_daily_volume_fraction=.1,
    ))
    allowed = run_swing_backtest(df, SwingConfig(
        min_prior_median_turnover_sek=0,
        max_daily_volume_fraction=.1,
    ))
    assert restricted.fills == ()
    assert restricted.liquidity_rejected_entries > 0
    assert any(fill.side == "BUY" for fill in allowed.fills)


def test_halt_does_not_create_phantom_mark_to_market_drawdown():
    df = bars()
    df.loc[225, ["open", "high", "low", "close", "volume"]] = [
        1.0, 1.0, 1.0, 1.0, 0
    ]
    result = run_swing_backtest(df)
    assert any(fill.side == "BUY" and fill.date < str(df.loc[225, "date"].date())
               for fill in result.fills)
    assert result.equity.iloc[225] == pytest.approx(result.equity.iloc[224])
    assert not any(f.date == str(df.loc[225, "date"].date()) for f in result.fills)


@pytest.mark.parametrize("kwargs", [
    {"min_prior_median_turnover_sek": -1},
    {"min_prior_median_turnover_sek": float("nan")},
    {"max_daily_volume_fraction": 0},
    {"max_daily_volume_fraction": .5},
    {"max_daily_volume_fraction": float("inf")},
])
def test_unsafe_liquidity_settings_fail_closed(kwargs):
    with pytest.raises(ValueError):
        SwingConfig(**kwargs)
