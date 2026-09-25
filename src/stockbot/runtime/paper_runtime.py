from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from stockbot.execution.paper import ExecutionResult, PaperExecutor
from stockbot.execution.safeguard import (
    ExecutionIntent,
    ExecutionMode,
    SafeguardGate,
    SafeguardState,
)
from stockbot.journal.trade_journal import AppendOnlyTradeJournal, DecisionEvent
from stockbot.opportunity.ranker import (
    Opportunity,
    OpportunityPolicy,
    RankedOpportunity,
    rank_opportunities,
)


@dataclass(frozen=True)
class PaperTradePlan:
    decision_id: str
    opportunity: Opportunity
    risk_fraction_of_equity: float
    notional_fraction_of_equity: float

    def __post_init__(self) -> None:
        if not self.decision_id.strip():
            raise ValueError("decision_id is required")
        if not 0.0 <= self.risk_fraction_of_equity < 1.0:
            raise ValueError("risk_fraction_of_equity must be in [0,1)")
        if self.notional_fraction_of_equity < 0.0:
            raise ValueError("notional_fraction_of_equity must be non-negative")


@dataclass(frozen=True)
class PaperRuntimeResult:
    ranked: tuple[RankedOpportunity, ...]
    executions: tuple[ExecutionResult, ...]
    accepted: int
    rejected: int


class PaperTradingRuntime:
    def __init__(
        self,
        journal_path: str | Path,
        *,
        opportunity_policy: OpportunityPolicy | None = None,
        safeguard: SafeguardGate | None = None,
    ) -> None:
        self.opportunity_policy = opportunity_policy or OpportunityPolicy()
        self.executor = PaperExecutor(safeguard or SafeguardGate())
        self.journal = AppendOnlyTradeJournal(journal_path)

    def run_cycle(
        self,
        plans: list[PaperTradePlan],
        state: SafeguardState,
        *,
        timestamp: datetime,
        regime: str,
        mode: ExecutionMode = ExecutionMode.PAPER,
    ) -> PaperRuntimeResult:
        by_symbol_source = {
            (plan.opportunity.symbol, plan.opportunity.source): plan
            for plan in plans
        }
        ranked = rank_opportunities(
            [plan.opportunity for plan in plans],
            self.opportunity_policy,
        )

        executions: list[ExecutionResult] = []
        accepted = 0
        rejected = 0
        rolling_open_risk = float(state.open_risk_usd)
        rolling_gross = float(state.gross_notional_usd)
        rolling_orders = int(state.orders_last_minute)

        for row in ranked:
            plan = by_symbol_source[
                (row.opportunity.symbol, row.opportunity.source)
            ]
            equity = float(state.equity_usd)
            intent = ExecutionIntent(
                decision_id=plan.decision_id,
                symbol=row.opportunity.symbol,
                mode=mode,
                estimated_risk_usd=equity * plan.risk_fraction_of_equity,
                notional_usd=equity * plan.notional_fraction_of_equity,
                expected_gross_edge_usd=(
                    row.opportunity.expected_return
                    * equity
                    * plan.notional_fraction_of_equity
                ),
                expected_cost_usd=(
                    row.opportunity.expected_cost
                    * equity
                    * plan.notional_fraction_of_equity
                ),
            )
            current_state = SafeguardState(
                equity_usd=state.equity_usd,
                equity_peak_usd=state.equity_peak_usd,
                daily_pnl_usd=state.daily_pnl_usd,
                open_risk_usd=rolling_open_risk,
                gross_notional_usd=rolling_gross,
                data_age_seconds=state.data_age_seconds,
                orders_last_minute=rolling_orders,
                kill_switch=state.kill_switch,
            )
            result = self.executor.submit(intent, current_state)
            executions.append(result)

            self.journal.append(
                DecisionEvent(
                    decision_id=plan.decision_id,
                    timestamp=timestamp,
                    symbol=row.opportunity.symbol,
                    strategy=row.opportunity.source,
                    regime=regime,
                    signal_score=max(
                        0.0,
                        min(1.0, 0.5 + row.net_edge * 10.0),
                    ),
                    confidence=row.opportunity.confidence,
                    expected_return=row.opportunity.expected_return,
                    expected_cost=row.opportunity.expected_cost,
                    expected_risk=plan.risk_fraction_of_equity,
                    approved=result.accepted,
                    reasons=result.safeguard.reasons,
                    metadata={
                        "rank": row.rank,
                        "net_edge": row.net_edge,
                        "mode": mode.value,
                    },
                )
            )

            if result.accepted:
                accepted += 1
                rolling_open_risk += intent.estimated_risk_usd
                rolling_gross += intent.notional_usd
                rolling_orders += 1
            else:
                rejected += 1

        return PaperRuntimeResult(
            ranked=tuple(ranked),
            executions=tuple(executions),
            accepted=accepted,
            rejected=rejected,
        )
