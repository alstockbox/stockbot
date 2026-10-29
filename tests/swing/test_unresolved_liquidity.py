"""Regression of unresolved liquidity risk near dataset end (synthetic only)."""
from __future__ import annotations
import pandas as pd
import pytest
from stockbot.swing.engine import SwingConfig, run_swing_backtest


def make_data():
    dates = pd.bdate_range("2025-01-02", periods=260)
    rows=[]
    for i,d in enumerate(dates):
        p=20+i*.11
        rows.append({"date":d,"symbol":"TEST","open":p-.04,
                     "high":p+.05,"low":p-.12,"close":p,"volume":10000})
    return pd.DataFrame(rows)


def test_end_of_data_never_fabricates_complete_exit_from_low_volume():
    data=make_data()
    for i in (258,259):
        data.loc[i,["open","high","low","close","volume"]]=[30.0,30.2,29.8,30.0,50]
    result=run_swing_backtest(data,SwingConfig(max_daily_volume_fraction=.02,
                                              execution_bps_each_side=0))
    assert any(f.side=="BUY" and f.date<str(data.loc[258,"date"].date())
               for f in result.fills)
    assert result.unresolved_exit_shares>0
    assert result.partial_exit_events>=1
    sell_shares_at_end=sum(f.quantity for f in result.fills if f.side=="SELL"
                           and f.date>=str(data.loc[258,"date"].date()))
    assert sell_shares_at_end<=2


def test_halted_day_zero_volume_does_not_reset_pending_exit():
    data=make_data()
    # Stop fires on 256 and tries to sell only 1 share.
    data.loc[256,["open","high","low","close","volume"]]=[30,30.2,29.8,30,50]
    data.loc[257,["open","high","low","close","volume"]]=[1,1,1,1,0]
    for i in (258,259):
        data.loc[i,["open","high","low","close","volume"]]=[29,29.2,28.8,29,50]
    result=run_swing_backtest(data,SwingConfig(max_daily_volume_fraction=.02,
                                              execution_bps_each_side=0))
    halted_date=str(data.loc[257,"date"].date())
    assert not any(f.date==halted_date for f in result.fills)
    assert result.equity.iloc[257] == pytest.approx(result.equity.iloc[256])
    assert any(f.reason=="EXIT_RETRY" and f.date>=str(data.loc[258,"date"].date())
               for f in result.fills)
    assert result.unresolved_exit_shares>0
