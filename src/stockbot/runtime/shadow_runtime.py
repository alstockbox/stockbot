from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from stockbot.data.live_models import AccountSnapshot, MarketQuote, SymbolSpec
from stockbot.data.live_validation import (
    LiveValidationConfig,
    validate_account,
    validate_quote,
    validate_symbol,
)
from stockbot.execution.safeguard import (
    ExecutionIntent,
    ExecutionMode,
    SafeguardGate,
    SafeguardState,
)
from stockbot.execution.shadow import ShadowExecutor, ShadowPosition
from stockbot.journal.trade_journal import AppendOnlyTradeJournal, DecisionEvent
from stockbot.opportunity.ranker import (
    Opportunity,
    OpportunityPolicy,
    RankedOpportunity,
    rank_opportunities,
)


class ReadOnlyMarketProvider(Protocol):
    def account_snapshot(self, *, now: datetime | None = None) -> AccountSnapshot: ...
    def quote(self, symbol: str) -> MarketQuote: ...
    def symbol_spec(self, symbol: str) -> SymbolSpec: ...


@dataclass(frozen=True)
class ShadowTradePlan:
    decision_id: str
    opportunity: Opportunity
    risk_fraction_of_equity: float
    notional_fraction_of_equity: float
    reward_risk: float = 2.0

    def __post_init__(self) -> None:
        if not self.decision_id.strip():
            raise ValueError("decision_id is required")
        if not 0.0 < self.risk_fraction_of_equity < 1.0:
            raise ValueError("risk_fraction_of_equity must be in (0,1)")
        if not 0.0 < self.notional_fraction_of_equity:
            raise ValueError("notional_fraction_of_equity must be positive")
        if self.reward_risk <= 0.0:
            raise ValueError("reward_risk must be positive")


@dataclass(frozen=True)
class ShadowRuntimeDecision:
    decision_id: str
    approved: bool
    reasons: tuple[str, ...]
    position: ShadowPosition | None = None


@dataclass(frozen=True)
class ShadowRuntimeResult:
    ranked: tuple[RankedOpportunity, ...]
    decisions: tuple[ShadowRuntimeDecision, ...]
    accepted: int
    rejected: int


class ShadowTradingRuntime:
    def __init__(
        self,
        provider: ReadOnlyMarketProvider,
        journal_path: str | Path,
        *,
        opportunity_policy: OpportunityPolicy | None = None,
        safeguard: SafeguardGate | None = None,
        validation: LiveValidationConfig | None = None,
        executor: ShadowExecutor | None = None,
    ) -> None:
        self.provider = provider
        self.opportunity_policy = opportunity_policy or OpportunityPolicy()
        self.safeguard = safeguard or SafeguardGate()
        self.validation = validation or LiveValidationConfig()
        self.executor = executor or ShadowExecutor()
        self.journal = AppendOnlyTradeJournal(journal_path)

    def _journal(
        self,
        plan: ShadowTradePlan,
        *,
        timestamp: datetime,
        regime: str,
        approved: bool,
        reasons: tuple[str, ...],
        expected_cost: float,
        metadata: dict[str, object],
    ) -> None:
        self.journal.append(
            DecisionEvent(
                decision_id=plan.decision_id,
                timestamp=timestamp,
                symbol=plan.opportunity.symbol,
                strategy=plan.opportunity.source,
                regime=regime,
                signal_score=max(
                    0.0,
                    min(1.0, 0.5 + float(plan.opportunity.expected_return) * 10.0),
                ),
                confidence=plan.opportunity.confidence,
                expected_return=plan.opportunity.expected_return,
                expected_cost=expected_cost,
                expected_risk=plan.risk_fraction_of_equity,
                approved=approved,
                reasons=reasons,
                metadata=metadata,
            )
        )

    def run_cycle(
        self,
        plans: list[ShadowTradePlan],
        *,
        timestamp: datetime,
        regime: str,
        equity_peak_usd: float | None = None,
        daily_pnl_usd: float = 0.0,
        orders_last_minute: int = 0,
        kill_switch: bool = False,
    ) -> ShadowRuntimeResult:
        existing_ids = {entry.event.decision_id for entry in self.journal.entries()}
        cycle_ids = set(existing_ids)
        decisions: list[ShadowRuntimeDecision] = []
        unseen_plans: list[ShadowTradePlan] = []

        for plan in plans:
            if plan.decision_id in cycle_ids:
                decisions.append(
                    ShadowRuntimeDecision(
                        plan.decision_id,
                        False,
                        ("DUPLICATE_DECISION_ID",),
                    )
                )
            else:
                cycle_ids.add(plan.decision_id)
                unseen_plans.append(plan)

        try:
            account = self.provider.account_snapshot(now=timestamp)
            account_check = validate_account(
                account,
                now=timestamp,
                config=self.validation,
            )
        except Exception:
            account = None
            account_check = None

        if account is None or account_check is None or not account_check.ok:
            reasons = (
                ("ACCOUNT_DATA_UNAVAILABLE",)
                if account_check is None
                else account_check.reasons
            )
            for plan in unseen_plans:
                self._journal(
                    plan,
                    timestamp=timestamp,
                    regime=regime,
                    approved=False,
                    reasons=reasons,
                    expected_cost=plan.opportunity.expected_cost,
                    metadata={"mode": ExecutionMode.SHADOW.value},
                )
                decisions.append(
                    ShadowRuntimeDecision(plan.decision_id, False, reasons)
                )
            return ShadowRuntimeResult(
                ranked=(),
                decisions=tuple(decisions),
                accepted=0,
                rejected=len(decisions),
            )

        adjusted: list[Opportunity] = []
        context: dict[int, tuple[ShadowTradePlan, MarketQuote]] = {}

        for plan in unseen_plans:
            try:
                quote = self.provider.quote(plan.opportunity.symbol)
                spec = self.provider.symbol_spec(plan.opportunity.symbol)
                quote_check = validate_quote(
                    quote,
                    now=timestamp,
                    config=self.validation,
                )
                symbol_check = validate_symbol(spec)
                reasons = quote_check.reasons + symbol_check.reasons
            except Exception:
                quote = None
                reasons = ("MARKET_DATA_UNAVAILABLE",)

            if quote is None or reasons:
                self._journal(
                    plan,
                    timestamp=timestamp,
                    regime=regime,
                    approved=False,
                    reasons=reasons,
                    expected_cost=plan.opportunity.expected_cost,
                    metadata={"mode": ExecutionMode.SHADOW.value},
                )
                decisions.append(
                    ShadowRuntimeDecision(plan.decision_id, False, reasons)
                )
                continue

            actual_cost = max(
                float(plan.opportunity.expected_cost),
                float(quote.spread_fraction),
            )
            live_opportunity = Opportunity(
                symbol=plan.opportunity.symbol,
                source=plan.opportunity.source,
                expected_return=plan.opportunity.expected_return,
                expected_cost=actual_cost,
                downside_risk=plan.opportunity.downside_risk,
                correlation_penalty=plan.opportunity.correlation_penalty,
                uncertainty=plan.opportunity.uncertainty,
                confidence=plan.opportunity.confidence,
            )
            adjusted.append(live_opportunity)
            context[id(live_opportunity)] = (plan, quote)

        ranked = rank_opportunities(adjusted, self.opportunity_policy)
        ranked_ids = {id(row.opportunity) for row in ranked}

        for opportunity in adjusted:
            if id(opportunity) in ranked_ids:
                continue
            plan, quote = context[id(opportunity)]
            reasons = ("BELOW_MINIMUM_NET_EDGE",)
            self._journal(
                plan,
                timestamp=timestamp,
                regime=regime,
                approved=False,
                reasons=reasons,
                expected_cost=opportunity.expected_cost,
                metadata={
                    "mode": ExecutionMode.SHADOW.value,
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "spread_fraction": quote.spread_fraction,
                },
            )
            decisions.append(
                ShadowRuntimeDecision(plan.decision_id, False, reasons)
            )

        peak = float(equity_peak_usd or account.equity)

        for row in ranked:
            plan, quote = context[id(row.opportunity)]
            equity = float(account.equity)
            risk_usd = equity * plan.risk_fraction_of_equity
            notional_usd = equity * plan.notional_fraction_of_equity
            intent = ExecutionIntent(
                decision_id=plan.decision_id,
                symbol=plan.opportunity.symbol,
                mode=ExecutionMode.SHADOW,
                estimated_risk_usd=risk_usd,
                notional_usd=notional_usd,
                expected_gross_edge_usd=(
                    float(row.opportunity.expected_return) * notional_usd
                ),
                expected_cost_usd=(
                    float(row.opportunity.expected_cost) * notional_usd
                ),
            )
            state = SafeguardState(
                equity_usd=equity,
                equity_peak_usd=peak,
                daily_pnl_usd=daily_pnl_usd,
                open_risk_usd=self.executor.open_risk_usd,
                gross_notional_usd=self.executor.gross_notional_usd,
                data_age_seconds=quote.age_seconds(timestamp),
                orders_last_minute=orders_last_minute,
                kill_switch=kill_switch,
            )
            safeguard = self.safeguard.evaluate(intent, state)

            position: ShadowPosition | None = None
            reasons = safeguard.reasons
            approved = safeguard.approved
            if approved:
                try:
                    position = self.executor.open_long(
                        decision_id=plan.decision_id,
                        quote=quote,
                        notional_usd=notional_usd,
                        estimated_risk_usd=risk_usd,
                        reward_risk=plan.reward_risk,
                    )
                except ValueError:
                    approved = False
                    reasons = ("SHADOW_EXECUTION_REJECTED",)

            metadata: dict[str, object] = {
                "mode": ExecutionMode.SHADOW.value,
                "rank": row.rank,
                "net_edge": row.net_edge,
                "bid": quote.bid,
                "ask": quote.ask,
                "spread_fraction": quote.spread_fraction,
            }
            if position is not None:
                metadata.update(
                    {
                        "entry_price": position.entry_price,
                        "stop_loss": position.stop_loss,
                        "take_profit": position.take_profit,
                        "notional_usd": position.notional_usd,
                    }
                )

            self._journal(
                plan,
                timestamp=timestamp,
                regime=regime,
                approved=approved,
                reasons=reasons,
                expected_cost=row.opportunity.expected_cost,
                metadata=metadata,
            )
            decisions.append(
                ShadowRuntimeDecision(
                    plan.decision_id,
                    approved,
                    reasons,
                    position,
                )
            )

        accepted = sum(1 for decision in decisions if decision.approved)
        return ShadowRuntimeResult(
            ranked=tuple(ranked),
            decisions=tuple(decisions),
            accepted=accepted,
            rejected=len(decisions) - accepted,
        )
