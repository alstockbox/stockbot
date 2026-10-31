from __future__ import annotations

import pandas as pd
import pytest

from stockbot.swing.engine import SwingConfig, run_swing_backtest
from stockbot.swing.walkforward import WalkForwardConfig, run_walkforward


def sample(n: int = 455) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({
        "date": dates, "symbol": ["TEST"] * n,
        "open": [60.0 + i * 0.09 for i in range(n)],
        "high": [61.0 + i * 0.09 for i in range(n)],
        "low": [59.0 + i * 0.09 for i in range(n)],
        "close": [60.3 + i * 0.09 for i in range(n)],
        "volume": [50000] * n,
    })


def short_config() -> WalkForwardConfig:
    return WalkForwardConfig(train_sessions=252, test_sessions=63,
                             min_folds=3, stop_candidates=(2.0, 2.5),
                             stress_execution_bps=(5.0, 30.0))


def test_walkforward_reports_complete_nonoverlapping_oos_without_orders():
    res = run_walkforward(sample(), short_config())
    assert res.mode == "WALK_FORWARD_BACKTEST_ONLY"
    assert res.fold_count == 3
    assert res.money_traded_sek == 0.0
    assert res.sufficient_activity_for_live_decision is False
    assert res.independent_benchmark_validated is False
    assert res.data_quality == "UNVERIFIED"
    for idx, fold in enumerate(res.folds):
        assert fold.train_end < fold.test_start <= fold.test_end
        assert fold.chosen_atr_stop_multiple in (2.0, 2.5)
        assert set(fold.cost_stress_returns_pct) == {"5bps", "30bps"}
        if idx:
            assert res.folds[idx - 1].test_end < fold.test_start
    # Each fold uses EXACTLY train_sessions, never an expanding window.
    all_dates = pd.bdate_range("2023-01-02", periods=455)
    for idx, fold in enumerate(res.folds):
        offset = idx * short_config().test_sessions
        assert fold.train_start == str(all_dates[offset].date())
        assert fold.train_end == str(all_dates[offset + 251].date())
    assert len(res.data_sha256) == 64
    assert res.total_hypothetical_fills >= 0


def test_no_trades_or_pnl_before_fold_first_trade_date():
    data = sample()
    start = data.iloc[300]["date"].strftime("%Y-%m-%d")
    res = run_swing_backtest(data, SwingConfig(first_trade_date=start))
    assert all(fill.date >= start for fill in res.fills)
    assert (res.equity.loc[res.equity.index < pd.Timestamp(start)] == 1000.0).all()
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        SwingConfig(first_trade_date="not-a-date")


def test_future_changes_cannot_change_first_fold_training_selection():
    original = sample()
    changed = original.copy()
    # Modify future TEST candles ONLY; training split is 252.
    for column in ("open", "high", "low", "close"):
        changed.loc[252:, column] = changed.loc[252:, column] * 0.85
    a = run_walkforward(original, short_config())
    b = run_walkforward(changed, short_config())
    assert a.folds[0].chosen_atr_stop_multiple == b.folds[0].chosen_atr_stop_multiple
    assert a.folds[0].training_net_return_pct == b.folds[0].training_net_return_pct


def test_walkforward_fails_closed_on_short_history_and_unsupported_configuration():
    with pytest.raises(ValueError, match="insufficient complete"):
        run_walkforward(sample(320), short_config())
    with pytest.raises(ValueError, match="small preregistered"):
        WalkForwardConfig(stop_candidates=tuple(range(1, 11)))
    with pytest.raises(ValueError, match="first_trade_date"):
        run_walkforward(sample(), WalkForwardConfig(
            train_sessions=252, test_sessions=63, min_folds=3,
            base=SwingConfig(first_trade_date="2024-01-01"),
        ))
