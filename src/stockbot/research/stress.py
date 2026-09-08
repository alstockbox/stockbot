from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import numpy as np
import pandas as pd

from stockbot.evaluation.metrics import performance_metrics


@dataclass(frozen=True)
class StressScenario:
    name: str
    loss_multiplier: float = 1.0
    gain_multiplier: float = 1.0
    extra_cost_bps_per_turnover: float = 0.0
    shock_every: int | None = None
    shock_size: float = 0.0

    def __post_init__(self) -> None:
        if self.loss_multiplier < 0 or self.gain_multiplier < 0:
            raise ValueError("multipliers must be non-negative")
        if self.extra_cost_bps_per_turnover < 0:
            raise ValueError("extra costs must be non-negative")
        if self.shock_every is not None and self.shock_every <= 0:
            raise ValueError("shock_every must be positive")
        if self.shock_size > 0:
            raise ValueError("shock_size must be zero or negative")


@dataclass(frozen=True)
class StressReport:
    scenario_metrics: dict[str, dict[str, float]]
    survival_rate: float
    median_sharpe: float
    worst_drawdown: float
    worst_cvar_95: float
    score: float


def default_stress_suite() -> tuple[StressScenario, ...]:
    return (
        StressScenario("losses_x1_25", loss_multiplier=1.25),
        StressScenario("losses_x1_50", loss_multiplier=1.50),
        StressScenario("costs_plus_10bps", extra_cost_bps_per_turnover=10.0),
        StressScenario("costs_plus_25bps", extra_cost_bps_per_turnover=25.0),
        StressScenario("periodic_gap_shock", shock_every=63, shock_size=-0.03),
        StressScenario(
            "combined_adverse",
            loss_multiplier=1.35,
            gain_multiplier=0.90,
            extra_cost_bps_per_turnover=15.0,
            shock_every=84,
            shock_size=-0.025,
        ),
    )


def _apply_scenario(
    returns: pd.Series,
    turnover: pd.Series,
    scenario: StressScenario,
) -> pd.Series:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()
    t = pd.Series(turnover, dtype=float).reindex(r.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    losses = r < 0
    gains = r > 0
    r.loc[losses] = r.loc[losses] * scenario.loss_multiplier
    r.loc[gains] = r.loc[gains] * scenario.gain_multiplier

    if scenario.extra_cost_bps_per_turnover:
        r = r - t.abs() * (scenario.extra_cost_bps_per_turnover / 10_000.0)

    if scenario.shock_every is not None and scenario.shock_size:
        positions = np.arange(len(r))
        mask = positions > 0
        mask &= positions % scenario.shock_every == 0
        if mask.any():
            r.iloc[np.where(mask)[0]] = r.iloc[np.where(mask)[0]] + scenario.shock_size

    return r.clip(lower=-0.999999)


def evaluate_stress_suite(
    returns: pd.Series,
    *,
    turnover: pd.Series | None = None,
    scenarios: tuple[StressScenario, ...] | list[StressScenario] | None = None,
    max_survival_drawdown: float = 0.40,
) -> StressReport:
    """Evaluate a realized strategy-return stream under deterministic adversarial shocks."""

    r = pd.Series(returns, dtype=float)
    t = pd.Series(0.0, index=r.index, dtype=float) if turnover is None else pd.Series(turnover, dtype=float)
    suite = tuple(scenarios or default_stress_suite())
    if not suite:
        raise ValueError("at least one stress scenario is required")

    scenario_metrics: dict[str, dict[str, float]] = {}
    survived = 0
    sharpes: list[float] = []
    drawdowns: list[float] = []
    cvars: list[float] = []

    for scenario in suite:
        stressed = _apply_scenario(r, t, scenario)
        metrics = performance_metrics(stressed)
        scenario_metrics[scenario.name] = metrics
        dd = float(metrics.get("max_drawdown", 1.0))
        cvar = float(metrics.get("cvar_95", 1.0))
        sharpe = float(metrics.get("sharpe", 0.0))
        drawdowns.append(dd)
        cvars.append(cvar)
        sharpes.append(sharpe)
        if dd <= max_survival_drawdown and float(metrics.get("cagr", -1.0)) > -0.20:
            survived += 1

    survival_rate = survived / len(suite)
    median_sharpe = float(median(sharpes)) if sharpes else 0.0
    worst_drawdown = max(drawdowns) if drawdowns else 1.0
    worst_cvar = max(cvars) if cvars else 1.0

    sharpe_component = float(np.clip((median_sharpe + 1.0) / 3.0, 0.0, 1.0))
    drawdown_component = float(np.clip(1.0 - worst_drawdown, 0.0, 1.0))
    tail_component = float(np.clip(1.0 - 5.0 * worst_cvar, 0.0, 1.0))
    score = float(np.clip(0.50 * survival_rate + 0.20 * sharpe_component + 0.20 * drawdown_component + 0.10 * tail_component, 0.0, 1.0))

    return StressReport(
        scenario_metrics=scenario_metrics,
        survival_rate=float(survival_rate),
        median_sharpe=median_sharpe,
        worst_drawdown=float(worst_drawdown),
        worst_cvar_95=float(worst_cvar),
        score=score,
    )
