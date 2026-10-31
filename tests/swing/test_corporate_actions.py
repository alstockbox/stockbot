"""Corporate actions and halted-session regression checks. Synthetic data only."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from stockbot.swing.corporate_actions import CorporateAction
from stockbot.swing.engine import SwingConfig, run_swing_backtest


def bars_with_trend(*, split_index: int | None = None, dividend_index: int | None = None):
    dates = pd.bdate_range("2025-01-02", periods=260)
    rows = []
    for i, day in enumerate(dates):
        p = 20.0 + i * 0.11
        if split_index is not None and i >= split_index:
            p /= 2
        if dividend_index is not None and i >= dividend_index:
            p -= 1.0
        rows.append({
            "date": day, "symbol": "AAA", "open": p - 0.04,
            "high": p + 0.05, "low": p - 0.15, "close": p, "volume": 300000,
        })
    return pd.DataFrame(rows), dates


def event(day: pd.Timestamp, kind: str, *, numerator=1, denominator=1,
          amount=0.0, pay_date=None):
    return CorporateAction(
        "AAA", day.date(), kind,
        (day - pd.Timedelta(days=7)).to_pydatetime().replace(tzinfo=timezone.utc),
        numerator=numerator, denominator=denominator,
        dividend_per_share_sek=amount, pay_date=pay_date,
    )


def test_split_adjusts_quantity_and_stop_without_manufacturing_loss():
    base, dates = bars_with_trend()
    split, _ = bars_with_trend(split_index=225)
    cfg = SwingConfig(execution_bps_each_side=0, commission_rate=0,
                      minimum_commission_sek=0)
    action = event(dates[225], "SPLIT", numerator=2, denominator=1)
    reference = run_swing_backtest(base, cfg)
    actual = run_swing_backtest(split, cfg, corporate_actions=(action,))
    assert any(f.side == "BUY" and f.date < str(dates[225].date()) for f in actual.fills)
    assert actual.corporate_action_events == (f"{dates[225].date()}:AAA:SPLIT",)
    # Same economics as a hypothetical continuously unsplit price history.
    assert actual.final_equity_sek == pytest.approx(reference.final_equity_sek, abs=1.0)
    assert actual.max_drawdown < 0.08


def test_dividend_entitlement_reinvests_only_after_payment_date():
    frame, dates = bars_with_trend(dividend_index=225)
    cfg = SwingConfig(execution_bps_each_side=0, commission_rate=0,
                      minimum_commission_sek=0)
    action = event(dates[225], "DIVIDEND", amount=1.0, pay_date=dates[231].date())
    with_cash = run_swing_backtest(frame, cfg, corporate_actions=(action,))
    without_cash = run_swing_backtest(frame, cfg)
    assert with_cash.dividend_cash_paid_sek > 0
    assert with_cash.dividend_receivable_sek == 0
    assert with_cash.final_equity_sek > without_cash.final_equity_sek
    assert with_cash.corporate_action_events == (f"{dates[225].date()}:AAA:DIVIDEND",)


def test_unpaid_dividend_is_receivable_not_spendable_cash():
    frame, dates = bars_with_trend(dividend_index=250)
    action = event(dates[250], "DIVIDEND", amount=1.0,
                   pay_date=dates[-1].date() + timedelta(days=90))
    result = run_swing_backtest(frame, corporate_actions=(action,))
    assert result.dividend_cash_paid_sek == 0.0
    assert result.dividend_receivable_sek > 0


def test_halted_sessions_do_not_fill_even_when_stop_crosses():
    frame, dates = bars_with_trend()
    frame.loc[225, "volume"] = 0.0
    frame.loc[225, "low"] = 5.0
    result = run_swing_backtest(frame)
    assert not any(f.date == str(dates[225].date()) for f in result.fills)


def test_fractional_split_without_cash_in_lieu_fails_closed():
    frame, dates = bars_with_trend(split_index=225)
    action = event(dates[225], "SPLIT", numerator=1001, denominator=1000)
    with pytest.raises(ValueError, match="fractional split"):
        run_swing_backtest(frame, corporate_actions=(action,))


def test_corporate_events_cannot_be_announced_after_exchange_open():
    _, dates = bars_with_trend()
    with pytest.raises(ValueError, match="announced before"):
        CorporateAction(
            symbol="AAA", ex_date=dates[225].date(), kind="SPLIT",
            announced_at=dates[225].to_pydatetime().replace(
                hour=10, tzinfo=timezone.utc),
            numerator=2, denominator=1,
        )


def test_duplicate_corporate_action_fails_before_simulation():
    frame, dates = bars_with_trend(split_index=225)
    action = event(dates[225], "SPLIT", numerator=2, denominator=1)
    with pytest.raises(ValueError, match="duplicate action"):
        run_swing_backtest(frame, corporate_actions=(action, action))
