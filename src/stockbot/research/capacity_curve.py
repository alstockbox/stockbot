from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from stockbot.research.liquidity_execution import (
    LiquidityExecutionConfig,
    LiquidityExecutionReport,
    simulate_liquidity_aware_execution,
)


@dataclass(frozen=True)
class CapacityPoint:
    capital: float
    score: float
    cagr: float
    sharpe: float
    max_drawdown: float
    partial_fill_fraction: float
    average_tracking_error: float
    average_participation: float
    cost_ratio: float
    score_retention: float
    usable: bool


@dataclass(frozen=True)
class CapacityCurveReport:
    points: tuple[CapacityPoint, ...]
    max_effective_capital: float | None
    first_break_capital: float | None
    baseline_capital: float
    baseline_score: float
    score_retention_at_max: float
    capacity_score: float


def evaluate_capacity_curve(
    bars: pd.DataFrame,
    predictions: pd.Series,
    *,
    capital_levels: tuple[float, ...] = (
        10_000.0,
        25_000.0,
        50_000.0,
        100_000.0,
        250_000.0,
        500_000.0,
        1_000_000.0,
        2_500_000.0,
        5_000_000.0,
    ),
    top_fraction: float = 0.30,
    weighting: str = "equal",
    base_config: LiquidityExecutionConfig | None = None,
    oos_coverage: float = 1.0,
    min_score_retention: float = 0.70,
    max_partial_fill_fraction: float = 0.25,
    max_tracking_error: float = 0.35,
    min_sharpe: float = 0.0,
) -> CapacityCurveReport:
    """Stress one prediction stream across increasing portfolio capital.

    The curve uses identical signals and execution assumptions at every level; only
    capital changes. A point is considered usable when liquidity degradation remains
    within configured limits. This estimates strategy capacity rather than assuming
    that backtest returns scale linearly with capital.
    """

    levels = tuple(sorted({float(value) for value in capital_levels}))
    if not levels or any(value <= 0.0 for value in levels):
        raise ValueError("capital_levels must contain positive values")
    if not 0.0 < min_score_retention <= 1.0:
        raise ValueError("min_score_retention must be in (0,1]")
    if not 0.0 <= max_partial_fill_fraction <= 1.0:
        raise ValueError("max_partial_fill_fraction must be in [0,1]")
    if max_tracking_error < 0.0:
        raise ValueError("max_tracking_error must be non-negative")

    config = base_config or LiquidityExecutionConfig(capital=levels[0])
    reports: list[tuple[float, LiquidityExecutionReport]] = []
    for capital in levels:
        report = simulate_liquidity_aware_execution(
            bars,
            predictions,
            top_fraction=top_fraction,
            weighting=weighting,
            config=replace(config, capital=capital),
            oos_coverage=oos_coverage,
        )
        reports.append((capital, report))

    baseline_score = max(1e-12, float(reports[0][1].score))
    points: list[CapacityPoint] = []
    for capital, report in reports:
        score_retention = float(np.clip(report.score / baseline_score, 0.0, 10.0))
        sharpe = float(report.metrics.get("sharpe", 0.0))
        usable = (
            score_retention >= min_score_retention
            and report.partial_fill_fraction <= max_partial_fill_fraction
            and report.average_tracking_error <= max_tracking_error
            and sharpe >= min_sharpe
        )
        points.append(
            CapacityPoint(
                capital=float(capital),
                score=float(report.score),
                cagr=float(report.metrics.get("cagr", 0.0)),
                sharpe=sharpe,
                max_drawdown=float(report.metrics.get("max_drawdown", 0.0)),
                partial_fill_fraction=float(report.partial_fill_fraction),
                average_tracking_error=float(report.average_tracking_error),
                average_participation=float(report.average_participation),
                cost_ratio=float(report.metrics.get("cost_ratio", 0.0)),
                score_retention=score_retention,
                usable=bool(usable),
            )
        )

    usable_points = [point for point in points if point.usable]
    max_effective_capital = max((point.capital for point in usable_points), default=None)
    first_break_capital = next((point.capital for point in points if not point.usable), None)
    score_retention_at_max = float(points[-1].score_retention)
    usable_fraction = len(usable_points) / len(points)
    breadth = (
        0.0
        if max_effective_capital is None
        else np.log10(max_effective_capital / levels[0] + 1.0)
        / max(np.log10(levels[-1] / levels[0] + 1.0), 1e-12)
    )
    capacity_score = float(
        np.clip(
            0.55 * usable_fraction
            + 0.30 * float(np.clip(breadth, 0.0, 1.0))
            + 0.15 * float(np.clip(score_retention_at_max, 0.0, 1.0)),
            0.0,
            1.0,
        )
    )

    return CapacityCurveReport(
        points=tuple(points),
        max_effective_capital=max_effective_capital,
        first_break_capital=first_break_capital,
        baseline_capital=float(levels[0]),
        baseline_score=float(reports[0][1].score),
        score_retention_at_max=score_retention_at_max,
        capacity_score=capacity_score,
    )
