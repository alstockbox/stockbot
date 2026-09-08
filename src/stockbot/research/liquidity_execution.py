from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from stockbot.arena.experiments import _signal_weights
from stockbot.data.panel import build_panel
from stockbot.evaluation.metrics import performance_metrics
from stockbot.research.objective import risk_adjusted_objective
from stockbot.research.stress import StressReport, evaluate_stress_suite


@dataclass(frozen=True)
class LiquidityExecutionConfig:
    capital: float = 100_000.0
    commission_bps: float = 1.0
    spread_bps: float = 4.0
    impact_bps_at_one_pct_adv: float = 10.0
    max_participation: float = 0.05
    adv_window: int = 20

    def __post_init__(self) -> None:
        if self.capital <= 0:
            raise ValueError("capital must be positive")
        if min(self.commission_bps, self.spread_bps, self.impact_bps_at_one_pct_adv) < 0:
            raise ValueError("execution costs cannot be negative")
        if not 0.0 < self.max_participation <= 1.0:
            raise ValueError("max_participation must be in (0,1]")
        if self.adv_window <= 0:
            raise ValueError("adv_window must be positive")


@dataclass(frozen=True)
class LiquidityExecutionReport:
    returns: pd.Series
    turnover_series: pd.Series
    cost_series: pd.Series
    metrics: dict[str, float]
    stress_report: StressReport
    score: float
    average_participation: float
    max_participation: float
    partial_fill_fraction: float
    average_tracking_error: float
    capacity_utilization: float


def simulate_liquidity_aware_execution(
    bars: pd.DataFrame,
    predictions: pd.Series,
    *,
    top_fraction: float = 0.30,
    weighting: str = "equal",
    config: LiquidityExecutionConfig | None = None,
    oos_coverage: float = 1.0,
) -> LiquidityExecutionReport:
    """Simulate one-day-lagged portfolio execution with ADV participation limits.

    ADV is shifted by one day, so fill capacity never uses the current day's final
    volume before the trade. Desired trades that exceed the participation cap are
    partially filled. Market impact follows a square-root participation curve.
    """

    cfg = config or LiquidityExecutionConfig()
    panel = build_panel(bars)
    close = panel["close"].unstack("symbol").sort_index().astype(float)
    volume = panel["volume"].unstack("symbol").sort_index().astype(float)
    asset_returns = close.pct_change().fillna(0.0)
    dollar_volume = close * volume
    adv = dollar_volume.rolling(cfg.adv_window, min_periods=cfg.adv_window).mean().shift(1)

    target = _signal_weights(predictions, top_fraction, weighting).reindex(close.index).fillna(0.0)
    desired_execution = target.shift(1).fillna(0.0)

    current = pd.Series(0.0, index=close.columns, dtype=float)
    realized_weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    daily_cost = pd.Series(0.0, index=close.index, dtype=float)
    daily_turnover = pd.Series(0.0, index=close.index, dtype=float)
    tracking_error = pd.Series(0.0, index=close.index, dtype=float)
    participation_observations: list[float] = []
    total_desired_notional = 0.0
    total_filled_notional = 0.0
    capacity_used = 0.0
    capacity_available = 0.0

    for dt in close.index:
        desired = desired_execution.loc[dt].fillna(0.0)
        delta = desired - current
        desired_notional = delta.abs() * cfg.capital
        available_adv = adv.loc[dt].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        max_notional = available_adv * cfg.max_participation

        fill_ratio = pd.Series(0.0, index=delta.index, dtype=float)
        tradable = desired_notional.gt(0.0) & max_notional.gt(0.0)
        fill_ratio.loc[tradable] = (
            max_notional.loc[tradable] / desired_notional.loc[tradable]
        ).clip(upper=1.0)
        actual_delta = delta * fill_ratio
        actual_notional = actual_delta.abs() * cfg.capital

        participation = pd.Series(0.0, index=delta.index, dtype=float)
        valid_adv = actual_notional.gt(0.0) & available_adv.gt(0.0)
        participation.loc[valid_adv] = actual_notional.loc[valid_adv] / available_adv.loc[valid_adv]
        if valid_adv.any():
            participation_observations.extend(participation.loc[valid_adv].tolist())

        impact_bps = pd.Series(0.0, index=delta.index, dtype=float)
        impact_bps.loc[valid_adv] = cfg.impact_bps_at_one_pct_adv * np.sqrt(
            participation.loc[valid_adv] / 0.01
        )
        total_cost_bps = cfg.commission_bps + cfg.spread_bps + impact_bps
        cost_return = float((actual_delta.abs() * total_cost_bps / 10_000.0).sum())

        current = current + actual_delta
        realized_weights.loc[dt] = current
        daily_cost.loc[dt] = cost_return
        daily_turnover.loc[dt] = float(actual_delta.abs().sum())
        tracking_error.loc[dt] = float((desired - current).abs().sum())

        total_desired_notional += float(desired_notional.sum())
        total_filled_notional += float(actual_notional.sum())
        capacity_used += float(actual_notional.sum())
        capacity_available += float(max_notional.where(desired_notional.gt(0.0), 0.0).sum())

    gross = (realized_weights * asset_returns).sum(axis=1)
    net = gross - daily_cost

    pred_dates = predictions.dropna().index.get_level_values("timestamp")
    if len(pred_dates):
        start = pred_dates.min()
        net = net.loc[net.index >= start]
        daily_cost = daily_cost.loc[daily_cost.index >= start]
        daily_turnover = daily_turnover.loc[daily_turnover.index >= start]
        tracking_error = tracking_error.loc[tracking_error.index >= start]
        realized_weights = realized_weights.loc[realized_weights.index >= start]

    metrics = performance_metrics(net)
    metrics["turnover"] = float(daily_turnover.sum())
    metrics["cost_ratio"] = float(daily_cost.sum())
    metrics["concentration"] = (
        float(realized_weights.pow(2).sum(axis=1).mean()) if len(realized_weights) else 0.0
    )
    monthly = (1.0 + net).resample("21B").prod() - 1.0 if len(net) else pd.Series(dtype=float)
    metrics["instability"] = float(
        np.clip(monthly.std(ddof=0) if len(monthly) > 1 else 0.0, 0.0, 1.0)
    )
    robustness = float(
        np.clip(
            (1.0 - float(metrics.get("max_drawdown", 1.0))) * (1.0 - metrics["instability"]),
            0.0,
            1.0,
        )
    )
    stress = evaluate_stress_suite(net, turnover=daily_turnover)
    score = risk_adjusted_objective(
        metrics,
        robustness=robustness,
        oos_coverage=oos_coverage,
        stress_score=stress.score,
    )

    average_participation = float(np.mean(participation_observations)) if participation_observations else 0.0
    max_participation_seen = float(max(participation_observations)) if participation_observations else 0.0
    partial_fill_fraction = (
        0.0
        if total_desired_notional <= 0.0
        else float(np.clip(1.0 - total_filled_notional / total_desired_notional, 0.0, 1.0))
    )
    capacity_utilization = (
        0.0
        if capacity_available <= 0.0
        else float(np.clip(capacity_used / capacity_available, 0.0, 1.0))
    )

    return LiquidityExecutionReport(
        returns=net,
        turnover_series=daily_turnover,
        cost_series=daily_cost,
        metrics={key: float(value) for key, value in metrics.items()},
        stress_report=stress,
        score=float(score),
        average_participation=average_participation,
        max_participation=max_participation_seen,
        partial_fill_fraction=partial_fill_fraction,
        average_tracking_error=float(tracking_error.mean()) if len(tracking_error) else 0.0,
        capacity_utilization=capacity_utilization,
    )
