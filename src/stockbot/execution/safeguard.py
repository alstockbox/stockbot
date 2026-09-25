from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class ExecutionMode(str, Enum):
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"


@dataclass(frozen=True)
class ExecutionIntent:
    decision_id: str
    symbol: str
    mode: ExecutionMode
    estimated_risk_usd: float
    notional_usd: float
    expected_gross_edge_usd: float
    expected_cost_usd: float

    def __post_init__(self) -> None:
        if not self.decision_id.strip():
            raise ValueError("decision_id is required")
        if not self.symbol.strip():
            raise ValueError("symbol is required")
        for name in (
            "estimated_risk_usd",
            "notional_usd",
            "expected_cost_usd",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(float(self.expected_gross_edge_usd)):
            raise ValueError("expected_gross_edge_usd must be finite")


@dataclass(frozen=True)
class SafeguardConfig:
    live_execution_enabled: bool = False
    max_order_risk_pct: float = 0.005
    max_total_open_risk_pct: float = 0.02
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.10
    max_data_age_seconds: float = 30.0
    max_gross_notional_multiple: float = 2.0
    max_orders_per_minute: int = 10
    minimum_net_edge_usd: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "max_order_risk_pct",
            "max_total_open_risk_pct",
            "max_daily_loss_pct",
            "max_drawdown_pct",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0,1)")
        if self.max_data_age_seconds <= 0.0:
            raise ValueError("max_data_age_seconds must be positive")
        if self.max_gross_notional_multiple <= 0.0:
            raise ValueError("max_gross_notional_multiple must be positive")
        if self.max_orders_per_minute <= 0:
            raise ValueError("max_orders_per_minute must be positive")


@dataclass(frozen=True)
class SafeguardState:
    equity_usd: float
    equity_peak_usd: float
    daily_pnl_usd: float = 0.0
    open_risk_usd: float = 0.0
    gross_notional_usd: float = 0.0
    data_age_seconds: float = 0.0
    orders_last_minute: int = 0
    kill_switch: bool = False

    def __post_init__(self) -> None:
        if self.equity_usd <= 0.0:
            raise ValueError("equity_usd must be positive")
        if self.equity_peak_usd <= 0.0:
            raise ValueError("equity_peak_usd must be positive")


@dataclass(frozen=True)
class SafeguardDecision:
    approved: bool
    reasons: tuple[str, ...]
    net_edge_usd: float
    order_risk_pct: float
    total_open_risk_pct_after: float


class SafeguardGate:
    def __init__(self, config: SafeguardConfig | None = None) -> None:
        self.config = config or SafeguardConfig()

    def evaluate(
        self,
        intent: ExecutionIntent,
        state: SafeguardState,
    ) -> SafeguardDecision:
        reasons: list[str] = []
        equity = max(float(state.equity_usd), 1e-12)
        peak = max(float(state.equity_peak_usd), equity, 1e-12)

        net_edge = float(intent.expected_gross_edge_usd) - float(intent.expected_cost_usd)
        order_risk_pct = float(intent.estimated_risk_usd) / equity
        total_risk_pct = (
            float(state.open_risk_usd) + float(intent.estimated_risk_usd)
        ) / equity
        drawdown = max(0.0, 1.0 - equity / peak)
        daily_loss_pct = max(0.0, -float(state.daily_pnl_usd) / equity)
        gross_after = float(state.gross_notional_usd) + float(intent.notional_usd)

        if state.kill_switch:
            reasons.append("KILL_SWITCH")
        if intent.mode is ExecutionMode.LIVE and not self.config.live_execution_enabled:
            reasons.append("LIVE_EXECUTION_DISABLED")
        if state.data_age_seconds > self.config.max_data_age_seconds:
            reasons.append("STALE_DATA")
        if state.orders_last_minute >= self.config.max_orders_per_minute:
            reasons.append("ORDER_RATE_LIMIT")
        if order_risk_pct > self.config.max_order_risk_pct:
            reasons.append("ORDER_RISK_LIMIT")
        if total_risk_pct > self.config.max_total_open_risk_pct:
            reasons.append("TOTAL_OPEN_RISK_LIMIT")
        if daily_loss_pct >= self.config.max_daily_loss_pct:
            reasons.append("DAILY_LOSS_LIMIT")
        if drawdown >= self.config.max_drawdown_pct:
            reasons.append("MAX_DRAWDOWN")
        if gross_after / equity > self.config.max_gross_notional_multiple:
            reasons.append("GROSS_EXPOSURE_LIMIT")
        if net_edge <= self.config.minimum_net_edge_usd:
            reasons.append("NO_POSITIVE_NET_EDGE")
        if intent.expected_gross_edge_usd > 0.0 and intent.expected_cost_usd >= intent.expected_gross_edge_usd:
            reasons.append("COST_CONSUMES_EDGE")

        return SafeguardDecision(
            approved=not reasons,
            reasons=tuple(reasons),
            net_edge_usd=net_edge,
            order_risk_pct=order_risk_pct,
            total_open_risk_pct_after=total_risk_pct,
        )
