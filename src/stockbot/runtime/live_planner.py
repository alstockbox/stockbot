from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from stockbot.data.live_models import MarketBar
from stockbot.research.live_signal import (
    LiveSignalAnalysis,
    LiveSignalConfig,
    LiveSignalEngine,
)
from stockbot.runtime.shadow_runtime import ShadowTradePlan


class HistoricalMarketProvider(Protocol):
    def bars(
        self,
        symbol: str,
        timeframe: Any = None,
        count: int = 250,
    ) -> list[MarketBar]: ...


@dataclass(frozen=True)
class LivePlannerConfig:
    bar_count: int = 250
    risk_fraction_of_equity: float = 0.003
    notional_fraction_of_equity: float = 0.25
    reward_risk: float = 2.0
    decision_namespace: str = "live-v1"

    def __post_init__(self) -> None:
        if self.bar_count < 80:
            raise ValueError("bar_count must be >= 80")
        if not 0.0 < self.risk_fraction_of_equity <= 0.005:
            raise ValueError("risk_fraction_of_equity must be in (0,0.005]")
        if not 0.0 < self.notional_fraction_of_equity <= 1.0:
            raise ValueError("notional_fraction_of_equity must be in (0,1]")
        if self.reward_risk <= 0.0:
            raise ValueError("reward_risk must be positive")
        if not self.decision_namespace.strip():
            raise ValueError("decision_namespace is required")


@dataclass(frozen=True)
class LivePlanResult:
    analysis: LiveSignalAnalysis
    plan: ShadowTradePlan | None
    last_bar_time: datetime


class LiveShadowPlanner:
    def __init__(
        self,
        provider: HistoricalMarketProvider,
        *,
        config: LivePlannerConfig | None = None,
        signal_config: LiveSignalConfig | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or LivePlannerConfig()
        self.engine = LiveSignalEngine(signal_config)

    def build(
        self,
        symbol: str,
        *,
        timeframe: Any = None,
    ) -> LivePlanResult:
        bars = self.provider.bars(
            symbol,
            timeframe=timeframe,
            count=self.config.bar_count,
        )
        if not bars:
            raise RuntimeError(f"NO_CLOSED_BARS:{symbol}")
        analysis = self.engine.analyze(symbol, bars)
        last_bar_time = max(bar.timestamp for bar in bars)
        opportunity = analysis.to_opportunity()
        if opportunity is None:
            return LivePlanResult(
                analysis=analysis,
                plan=None,
                last_bar_time=last_bar_time,
            )

        decision_id = (
            f"{self.config.decision_namespace}:"
            f"{symbol}:{int(last_bar_time.timestamp())}"
        )
        plan = ShadowTradePlan(
            decision_id=decision_id,
            opportunity=opportunity,
            risk_fraction_of_equity=self.config.risk_fraction_of_equity,
            notional_fraction_of_equity=self.config.notional_fraction_of_equity,
            reward_risk=self.config.reward_risk,
            market_regime=analysis.regime.value,
        )
        return LivePlanResult(
            analysis=analysis,
            plan=plan,
            last_bar_time=last_bar_time,
        )
