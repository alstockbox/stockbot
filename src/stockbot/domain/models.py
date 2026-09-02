from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd


class MarketRegime(str, Enum):
    BULL_TREND = "bull_trend"
    BEAR_STRESS = "bear_stress"
    NEUTRAL_CHOP = "neutral_chop"


@dataclass(frozen=True)
class Signal:
    symbol: str
    score: float
    source: str
    timestamp: datetime
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("signal score must be in [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("signal confidence must be in [0, 1]")
        if not self.symbol:
            raise ValueError("symbol is required")
        if not self.source:
            raise ValueError("source is required")


@dataclass(frozen=True)
class TargetPosition:
    symbol: str
    weight: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("V0 target weight must be in [0, 1]")
        if not self.symbol:
            raise ValueError("symbol is required")


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    target: TargetPosition
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    turnover: pd.Series
    costs: pd.Series
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return float(self.costs.sum())

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1])


@dataclass(frozen=True)
class ResearchRun:
    leaderboard: list[dict[str, Any]]
    ensemble: BacktestResult
    latest_regime: MarketRegime
    latest_signal: float
    hypotheses: list[Any]
    champion_name: str | None
    ml_oos_samples: int
