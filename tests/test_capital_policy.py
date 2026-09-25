import pytest

from stockbot.portfolio.capital_policy import CapitalState, close_month


def test_monthly_profit_withdraws_twenty_percent_above_high_water_mark():
    result = close_month(CapitalState(466.0, 466.0), 500.0)
    assert result.gross_profit_above_hwm == pytest.approx(34.0)
    assert result.withdrawal == pytest.approx(6.8)
    assert result.retained_profit == pytest.approx(27.2)
    assert result.ending_equity_after_withdrawal == pytest.approx(493.2)
    assert result.next_high_water_mark == pytest.approx(493.2)


def test_losing_month_has_no_withdrawal_and_preserves_high_water_mark():
    result = close_month(CapitalState(466.0, 466.0), 450.0)
    assert result.withdrawal == 0.0
    assert result.next_high_water_mark == pytest.approx(466.0)


def test_deposit_is_not_misclassified_as_profit():
    result = close_month(
        CapitalState(466.0, 466.0),
        580.0,
        net_contribution=100.0,
    )
    assert result.gross_profit_above_hwm == pytest.approx(14.0)
    assert result.withdrawal == pytest.approx(2.8)
