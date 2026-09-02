from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from stockbot.costs.model import LinearCostModel
from stockbot.domain.models import BacktestResult, TargetPosition
from stockbot.evaluation.metrics import performance_metrics
from stockbot.risk.engine import RiskEngine, RiskState


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    max_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0,1]")


def run_backtest(
    frame: pd.DataFrame,
    signal_series: pd.Series,
    config: BacktestConfig | None = None,
    benchmark_returns: pd.Series | None = None,
    risk_engine: RiskEngine | None = None,
) -> BacktestResult:
    config = config or BacktestConfig()
    if "close" not in frame.columns:
        raise ValueError("frame must contain close")
    if not frame.index.equals(signal_series.index):
        signal_series = signal_series.reindex(frame.index)

    close = frame["close"].astype(float)
    asset_returns = close.pct_change().fillna(0.0)
    requested = (
        signal_series.astype(float)
        .clip(0.0, config.max_weight)
        .fillna(0.0)
        .shift(1)
        .fillna(0.0)
    )
    prior_vol = (
        asset_returns.shift(1)
        .rolling(20, min_periods=5)
        .std(ddof=0)
        .mul(math.sqrt(252))
        .fillna(0.0)
    )

    cost_model = LinearCostModel(config.commission_bps, config.slippage_bps)

    equity_values: list[float] = []
    net_return_values: list[float] = []
    position_values: list[float] = []
    turnover_values: list[float] = []
    cost_values: list[float] = []

    equity = float(config.initial_capital)
    equity_peak = equity
    previous_position = 0.0
    previous_net_return = 0.0

    for i, timestamp in enumerate(frame.index):
        desired_weight = float(requested.iloc[i])

        if risk_engine is not None:
            decision = risk_engine.evaluate(
                TargetPosition("ASSET", desired_weight),
                RiskState(
                    equity=equity,
                    equity_peak=equity_peak,
                    daily_pnl_pct=previous_net_return,
                    data_age_seconds=0.0,
                    realized_volatility=float(prior_vol.iloc[i]),
                ),
            )
            desired_weight = decision.target.weight if decision.approved else 0.0

        turnover = abs(desired_weight - previous_position)
        cost_rate = cost_model.rate_for_turnover(turnover)
        gross_return = desired_weight * float(asset_returns.iloc[i])
        net_return = gross_return - cost_rate
        cost_dollars = equity * cost_rate

        equity *= 1.0 + net_return
        equity_peak = max(equity_peak, equity)

        position_values.append(desired_weight)
        turnover_values.append(turnover)
        cost_values.append(cost_dollars)
        net_return_values.append(net_return)
        equity_values.append(equity)

        previous_position = desired_weight
        previous_net_return = net_return

    positions = pd.Series(position_values, index=frame.index, dtype=float, name="position")
    turnover = pd.Series(turnover_values, index=frame.index, dtype=float, name="turnover")
    costs = pd.Series(cost_values, index=frame.index, dtype=float, name="cost")
    net_returns = pd.Series(net_return_values, index=frame.index, dtype=float, name="return")
    equity_curve = pd.Series(equity_values, index=frame.index, dtype=float, name="equity")

    metrics = performance_metrics(net_returns, benchmark_returns)
    metrics["turnover"] = float(turnover.sum())
    metrics["cost_ratio"] = float(
        sum(cost_model.rate_for_turnover(value) for value in turnover_values)
    )

    return BacktestResult(
        equity_curve=equity_curve,
        returns=net_returns,
        positions=positions,
        turnover=turnover,
        costs=costs,
        metrics=metrics,
    )
