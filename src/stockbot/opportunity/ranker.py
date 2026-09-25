from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Opportunity:
    symbol: str
    source: str
    expected_return: float
    expected_cost: float
    downside_risk: float
    correlation_penalty: float
    uncertainty: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol is required")
        if not self.source.strip():
            raise ValueError("source is required")
        for name in ("expected_cost", "downside_risk", "correlation_penalty", "uncertainty"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        if not math.isfinite(float(self.expected_return)):
            raise ValueError("expected_return must be finite")


@dataclass(frozen=True)
class OpportunityPolicy:
    risk_penalty: float = 1.0
    correlation_penalty: float = 1.0
    uncertainty_penalty: float = 1.0
    minimum_net_edge: float = 0.0
    max_opportunities: int | None = None

    def __post_init__(self) -> None:
        if min(self.risk_penalty, self.correlation_penalty, self.uncertainty_penalty) < 0.0:
            raise ValueError("penalty weights must be non-negative")
        if self.max_opportunities is not None and self.max_opportunities <= 0:
            raise ValueError("max_opportunities must be positive")


@dataclass(frozen=True)
class RankedOpportunity:
    opportunity: Opportunity
    net_edge: float
    rank: int


def expected_net_edge(
    opportunity: Opportunity,
    policy: OpportunityPolicy | None = None,
) -> float:
    policy = policy or OpportunityPolicy()
    gross_after_cost = float(opportunity.expected_return) - float(opportunity.expected_cost)
    risk_adjusted = (
        gross_after_cost
        - policy.risk_penalty * float(opportunity.downside_risk)
        - policy.correlation_penalty * float(opportunity.correlation_penalty)
        - policy.uncertainty_penalty * float(opportunity.uncertainty)
    )
    return float(opportunity.confidence) * risk_adjusted


def rank_opportunities(
    opportunities: list[Opportunity],
    policy: OpportunityPolicy | None = None,
) -> list[RankedOpportunity]:
    policy = policy or OpportunityPolicy()
    scored = [
        (opportunity, expected_net_edge(opportunity, policy))
        for opportunity in opportunities
    ]
    qualified = [
        row for row in scored if row[1] > float(policy.minimum_net_edge)
    ]
    qualified.sort(
        key=lambda row: (row[1], row[0].confidence, row[0].expected_return),
        reverse=True,
    )
    if policy.max_opportunities is not None:
        qualified = qualified[: policy.max_opportunities]
    return [
        RankedOpportunity(opportunity=row[0], net_edge=row[1], rank=index + 1)
        for index, row in enumerate(qualified)
    ]
