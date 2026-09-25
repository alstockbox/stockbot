from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradeCounterfactual:
    trade_id: str
    realized_net_return: float
    alternative_policy_return: float
    alternative_available_at_decision: bool
    execution_cost_return: float = 0.0


@dataclass(frozen=True)
class RegretSummary:
    observations: int
    total_regret: float
    mean_regret: float
    missed_gain: float
    avoidable_loss: float
    execution_cost_drag: float


def analyze_regret(observations: list[TradeCounterfactual]) -> RegretSummary:
    if any(not row.alternative_available_at_decision for row in observations):
        raise ValueError("counterfactuals must only use information available at decision time")

    regrets = [
        max(0.0, float(row.alternative_policy_return) - float(row.realized_net_return))
        for row in observations
    ]
    missed_gain = sum(
        max(0.0, float(row.alternative_policy_return))
        for row in observations
        if float(row.realized_net_return) == 0.0
    )
    avoidable_loss = sum(
        -float(row.realized_net_return)
        for row in observations
        if float(row.realized_net_return) < 0.0 and float(row.alternative_policy_return) >= 0.0
    )
    execution_cost_drag = sum(max(0.0, float(row.execution_cost_return)) for row in observations)
    total = float(sum(regrets))
    count = len(observations)
    return RegretSummary(
        observations=count,
        total_regret=total,
        mean_regret=total / count if count else 0.0,
        missed_gain=float(missed_gain),
        avoidable_loss=float(avoidable_loss),
        execution_cost_drag=float(execution_cost_drag),
    )
