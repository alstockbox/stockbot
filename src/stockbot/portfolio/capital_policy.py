from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapitalPolicy:
    withdrawal_rate: float = 0.20

    def __post_init__(self) -> None:
        if not 0.0 <= self.withdrawal_rate < 1.0:
            raise ValueError("withdrawal_rate must be in [0,1)")


@dataclass(frozen=True)
class CapitalState:
    equity: float
    high_water_mark: float

    def __post_init__(self) -> None:
        if self.equity < 0.0 or self.high_water_mark < 0.0:
            raise ValueError("capital values must be non-negative")


@dataclass(frozen=True)
class MonthClose:
    gross_profit_above_hwm: float
    withdrawal: float
    retained_profit: float
    ending_equity_after_withdrawal: float
    next_high_water_mark: float


def close_month(
    state: CapitalState,
    ending_equity_before_withdrawal: float,
    *,
    net_contribution: float = 0.0,
    policy: CapitalPolicy | None = None,
) -> MonthClose:
    policy = policy or CapitalPolicy()
    ending = float(ending_equity_before_withdrawal)
    if ending < 0.0:
        raise ValueError("ending equity must be non-negative")

    adjusted_hwm = max(0.0, float(state.high_water_mark) + float(net_contribution))
    profit = max(0.0, ending - adjusted_hwm)
    withdrawal = profit * policy.withdrawal_rate
    retained = profit - withdrawal
    after_withdrawal = ending - withdrawal
    next_hwm = max(adjusted_hwm, after_withdrawal)

    return MonthClose(
        gross_profit_above_hwm=profit,
        withdrawal=withdrawal,
        retained_profit=retained,
        ending_equity_after_withdrawal=after_withdrawal,
        next_high_water_mark=next_hwm,
    )
