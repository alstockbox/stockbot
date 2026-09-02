from __future__ import annotations

from dataclasses import dataclass

from stockbot.domain.models import RiskDecision, TargetPosition


@dataclass(frozen=True)
class RiskConfig:
    max_position_weight: float = 1.0
    max_drawdown: float = 0.10
    max_daily_loss: float = 0.04
    max_data_age_seconds: float = 300.0
    abnormal_volatility: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("max_position_weight must be in (0,1]")
        if not 0 < self.max_drawdown < 1:
            raise ValueError("max_drawdown must be in (0,1)")
        if not 0 < self.max_daily_loss < 1:
            raise ValueError("max_daily_loss must be in (0,1)")


@dataclass(frozen=True)
class RiskState:
    equity: float = 100.0
    equity_peak: float = 100.0
    daily_pnl_pct: float = 0.0
    data_age_seconds: float = 0.0
    realized_volatility: float = 0.0
    kill_switch: bool = False


class RiskEngine:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def evaluate(self, target: TargetPosition, state: RiskState) -> RiskDecision:
        reasons: list[str] = []
        hard_reject = False

        if state.kill_switch:
            reasons.append("KILL_SWITCH")
            hard_reject = True
        if state.data_age_seconds > self.config.max_data_age_seconds:
            reasons.append("STALE_DATA")
            hard_reject = True
        if state.daily_pnl_pct <= -self.config.max_daily_loss:
            reasons.append("DAILY_LOSS_LIMIT")
            hard_reject = True

        peak = max(float(state.equity_peak), 1e-12)
        drawdown = max(0.0, 1.0 - float(state.equity) / peak)
        if drawdown >= self.config.max_drawdown:
            reasons.append("MAX_DRAWDOWN")
            hard_reject = True

        if state.realized_volatility >= self.config.abnormal_volatility:
            reasons.append("ABNORMAL_VOLATILITY")
            hard_reject = True

        if hard_reject:
            return RiskDecision(False, TargetPosition(target.symbol, 0.0), tuple(reasons))

        approved_weight = min(target.weight, self.config.max_position_weight)
        if approved_weight < target.weight:
            reasons.append("POSITION_CAPPED")
        return RiskDecision(True, TargetPosition(target.symbol, approved_weight), tuple(reasons))
