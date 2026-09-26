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
from stockbot.execution.shadow import ShadowExecutor, ShadowExit, ShadowPosition
from stockbot.execution.shadow_state import ShadowPortfolioState, ShadowStateStore
from stockbot.journal.shadow_lifecycle import (
    AppendOnlyShadowLifecycleJournal,
    ShadowLifecycleEvent,
)
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
    market_regime: str | None = None

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


@dataclass(frozen=True)
class ShadowMarkIssue:
    decision_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ShadowRefreshResult:
    complete: bool
    exits: tuple[ShadowExit, ...]
    issues: tuple[ShadowMarkIssue, ...]
    unrealized_pnl_usd: float
    shadow_equity_usd: float
    equity_peak_usd: float


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
        state_path: str | Path | None = None,
        lifecycle_path: str | Path | None = None,
    ) -> None:
        self.provider = provider
        self.opportunity_policy = opportunity_policy or OpportunityPolicy()
        self.safeguard = safeguard or SafeguardGate()
        self.validation = validation or LiveValidationConfig()

        journal_file = Path(journal_path)
        self.journal = AppendOnlyTradeJournal(journal_file)
        self.state_store = ShadowStateStore(
            state_path
            or journal_file.with_name(journal_file.name + ".state.json")
        )
        self.lifecycle = AppendOnlyShadowLifecycleJournal(
            lifecycle_path
            or journal_file.with_name(journal_file.name + ".lifecycle.jsonl")
        )

        restored = self.state_store.load()
        if executor is None:
            self.executor = ShadowExecutor(
                restored.positions if restored is not None else None
            )
        else:
            self.executor = executor
            if restored is not None:
                restored_ids = {position.decision_id for position in restored.positions}
                executor_ids = {position.decision_id for position in executor.positions}
                if restored_ids != executor_ids:
                    raise ValueError(
                        "executor positions do not match persisted shadow state"
                    )
        self._portfolio_state = restored

    @property
    def portfolio_state(self) -> ShadowPortfolioState | None:
        return self._portfolio_state

    def _ensure_state(
        self,
        *,
        base_equity_usd: float,
        timestamp: datetime,
    ) -> ShadowPortfolioState:
        if self._portfolio_state is None:
            state = ShadowPortfolioState(
                base_equity_usd=float(base_equity_usd),
                realized_pnl_usd=0.0,
                equity_peak_usd=float(base_equity_usd),
                positions=self.executor.positions,
                updated_at=timestamp,
            )
            self.state_store.save(state)
            self._portfolio_state = state
        return self._portfolio_state

    def _persist_state(
        self,
        *,
        timestamp: datetime,
        realized_delta_usd: float = 0.0,
        equity_peak_usd: float | None = None,
    ) -> ShadowPortfolioState:
        if self._portfolio_state is None:
            raise RuntimeError("SHADOW_STATE_NOT_INITIALIZED")
        state = ShadowPortfolioState(
            base_equity_usd=self._portfolio_state.base_equity_usd,
            realized_pnl_usd=(
                self._portfolio_state.realized_pnl_usd
                + float(realized_delta_usd)
            ),
            equity_peak_usd=float(
                self._portfolio_state.equity_peak_usd
                if equity_peak_usd is None
                else equity_peak_usd
            ),
            positions=self.executor.positions,
            updated_at=timestamp,
        )
        self.state_store.save(state)
        self._portfolio_state = state
        return state

    def daily_realized_pnl(self, timestamp: datetime) -> float:
        return self.lifecycle.realized_pnl_for_date(timestamp.date())

    def refresh_positions(self, *, timestamp: datetime) -> ShadowRefreshResult:
        if self._portfolio_state is None:
            raise RuntimeError("SHADOW_STATE_NOT_INITIALIZED")

        exits: list[ShadowExit] = []
        issues: list[ShadowMarkIssue] = []
        unrealized = 0.0
        realized_delta = 0.0

        for position in tuple(self.executor.positions):
            try:
                quote = self.provider.quote(position.symbol)
                check = validate_quote(
                    quote,
                    now=timestamp,
                    config=self.validation,
                )
                if not check.ok:
                    issues.append(
                        ShadowMarkIssue(position.decision_id, check.reasons)
                    )
                    continue
            except Exception:
                issues.append(
                    ShadowMarkIssue(
                        position.decision_id,
                        ("MARKET_DATA_UNAVAILABLE",),
                    )
                )
                continue

            exit_result = self.executor.maybe_close(
                position.decision_id,
                quote,
            )
            if exit_result is not None:
                exits.append(exit_result)
                realized_delta += exit_result.realized_pnl_usd
                self.lifecycle.append(
                    ShadowLifecycleEvent(
                        event_id=f"{position.decision_id}:CLOSE",
                        decision_id=position.decision_id,
                        event_type="CLOSE",
                        timestamp=exit_result.closed_at,
                        symbol=position.symbol,
                        price=exit_result.exit_price,
                        pnl_usd=exit_result.realized_pnl_usd,
                        metadata={"reason": exit_result.reason},
                    )
                )
            else:
                unrealized += self.executor.unrealized_pnl(
                    position.decision_id,
                    quote,
                )

        prior = self._portfolio_state
        projected_realized = prior.realized_pnl_usd + realized_delta
        shadow_equity = (
            prior.base_equity_usd
            + projected_realized
            + unrealized
        )
        peak = prior.equity_peak_usd
        if not issues:
            peak = max(peak, shadow_equity)

        self._persist_state(
            timestamp=timestamp,
            realized_delta_usd=realized_delta,
            equity_peak_usd=peak,
        )

        return ShadowRefreshResult(
            complete=not issues,
            exits=tuple(exits),
            issues=tuple(issues),
            unrealized_pnl_usd=float(unrealized),
            shadow_equity_usd=float(shadow_equity),
            equity_peak_usd=float(peak),
        )

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
                regime=plan.market_regime or regime,
                signal_score=max(
                    0.0,
                    min(
                        1.0,
                        0.5
                        + float(plan.opportunity.expected_return) * 10.0,
                    ),
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

    def _reject_plans(
        self,
        plans: list[ShadowTradePlan],
        *,
        timestamp: datetime,
        regime: str,
        reasons: tuple[str, ...],
        decisions: list[ShadowRuntimeDecision],
        metadata: dict[str, object] | None = None,
    ) -> None:
        for plan in plans:
            self._journal(
                plan,
                timestamp=timestamp,
                regime=regime,
                approved=False,
                reasons=reasons,
                expected_cost=plan.opportunity.expected_cost,
                metadata={
                    "mode": ExecutionMode.SHADOW.value,
                    **(metadata or {}),
                },
            )
            decisions.append(
                ShadowRuntimeDecision(plan.decision_id, False, reasons)
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
        existing_ids = {
            entry.event.decision_id for entry in self.journal.entries()
        }
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
            self._reject_plans(
                unseen_plans,
                timestamp=timestamp,
                regime=regime,
                reasons=reasons,
                decisions=decisions,
            )
            return ShadowRuntimeResult(
                ranked=(),
                decisions=tuple(decisions),
                accepted=0,
                rejected=len(decisions),
            )

        self._ensure_state(
            base_equity_usd=float(account.equity),
            timestamp=timestamp,
        )
        refresh = self.refresh_positions(timestamp=timestamp)

        if not refresh.complete:
            self._reject_plans(
                unseen_plans,
                timestamp=timestamp,
                regime=regime,
                reasons=("PORTFOLIO_MARK_UNAVAILABLE",),
                decisions=decisions,
                metadata={
                    "mark_issues": [
                        {
                            "decision_id": issue.decision_id,
                            "reasons": list(issue.reasons),
                        }
                        for issue in refresh.issues
                    ]
                },
            )
            return ShadowRuntimeResult(
                ranked=(),
                decisions=tuple(decisions),
                accepted=0,
                rejected=len(decisions),
            )

        if refresh.shadow_equity_usd <= 0.0:
            self._reject_plans(
                unseen_plans,
                timestamp=timestamp,
                regime=regime,
                reasons=("SHADOW_EQUITY_NONPOSITIVE",),
                decisions=decisions,
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

        ranked = rank_opportunities(
            adjusted,
            self.opportunity_policy,
        )
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

        effective_equity = min(
            float(account.equity),
            float(refresh.shadow_equity_usd),
        )
        peak = float(
            max(
                refresh.equity_peak_usd,
                equity_peak_usd or 0.0,
                effective_equity,
            )
        )
        shadow_daily_pnl = self.daily_realized_pnl(timestamp)
        effective_daily_pnl = min(
            float(daily_pnl_usd),
            float(shadow_daily_pnl),
        )

        for row in ranked:
            plan, quote = context[id(row.opportunity)]
            risk_usd = effective_equity * plan.risk_fraction_of_equity
            notional_usd = (
                effective_equity * plan.notional_fraction_of_equity
            )
            intent = ExecutionIntent(
                decision_id=plan.decision_id,
                symbol=plan.opportunity.symbol,
                mode=ExecutionMode.SHADOW,
                estimated_risk_usd=risk_usd,
                notional_usd=notional_usd,
                expected_gross_edge_usd=(
                    float(row.opportunity.expected_return)
                    * notional_usd
                ),
                expected_cost_usd=(
                    float(row.opportunity.expected_cost)
                    * notional_usd
                ),
            )
            state = SafeguardState(
                equity_usd=effective_equity,
                equity_peak_usd=peak,
                daily_pnl_usd=effective_daily_pnl,
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
                    self._persist_state(timestamp=timestamp)
                    self.lifecycle.append(
                        ShadowLifecycleEvent(
                            event_id=f"{plan.decision_id}:OPEN",
                            decision_id=plan.decision_id,
                            event_type="OPEN",
                            timestamp=position.opened_at,
                            symbol=position.symbol,
                            price=position.entry_price,
                            pnl_usd=0.0,
                            metadata={
                                "stop_loss": position.stop_loss,
                                "take_profit": position.take_profit,
                                "notional_usd": position.notional_usd,
                                "estimated_risk_usd": (
                                    position.estimated_risk_usd
                                ),
                            },
                        )
                    )
                except Exception:
                    if position is not None:
                        try:
                            self.executor.discard(plan.decision_id)
                            self._persist_state(timestamp=timestamp)
                        except Exception:
                            pass
                    position = None
                    approved = False
                    reasons = ("SHADOW_PERSISTENCE_FAILED",)

            metadata: dict[str, object] = {
                "mode": ExecutionMode.SHADOW.value,
                "rank": row.rank,
                "net_edge": row.net_edge,
                "bid": quote.bid,
                "ask": quote.ask,
                "spread_fraction": quote.spread_fraction,
                "shadow_equity_usd": effective_equity,
                "shadow_daily_pnl_usd": shadow_daily_pnl,
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

        accepted = sum(
            1 for decision in decisions if decision.approved
        )
        return ShadowRuntimeResult(
            ranked=tuple(ranked),
            decisions=tuple(decisions),
            accepted=accepted,
            rejected=len(decisions) - accepted,
        )
