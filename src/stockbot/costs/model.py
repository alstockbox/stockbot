from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearCostModel:
    commission_bps: float = 1.0
    slippage_bps: float = 2.0

    @property
    def total_bps(self) -> float:
        return max(0.0, self.commission_bps) + max(0.0, self.slippage_bps)

    def estimate(self, notional: float, turnover: float = 1.0) -> float:
        return abs(float(notional)) * abs(float(turnover)) * self.total_bps / 10_000.0

    def rate_for_turnover(self, turnover: float) -> float:
        return abs(float(turnover)) * self.total_bps / 10_000.0
