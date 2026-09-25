from __future__ import annotations

from dataclasses import dataclass

from stockbot.execution.safeguard import (
    ExecutionIntent,
    ExecutionMode,
    SafeguardDecision,
    SafeguardGate,
    SafeguardState,
)


@dataclass(frozen=True)
class PaperFill:
    decision_id: str
    symbol: str
    notional_usd: float
    estimated_risk_usd: float


@dataclass(frozen=True)
class ExecutionResult:
    accepted: bool
    safeguard: SafeguardDecision
    fill: PaperFill | None = None


class PaperExecutor:
    def __init__(self, safeguard: SafeguardGate | None = None) -> None:
        self.safeguard = safeguard or SafeguardGate()
        self._seen_decision_ids: set[str] = set()

    def submit(self, intent: ExecutionIntent, state: SafeguardState) -> ExecutionResult:
        if intent.mode is ExecutionMode.LIVE:
            decision = self.safeguard.evaluate(intent, state)
            return ExecutionResult(False, decision, None)

        if intent.decision_id in self._seen_decision_ids:
            duplicate = SafeguardDecision(
                approved=False,
                reasons=("DUPLICATE_DECISION_ID",),
                net_edge_usd=float(intent.expected_gross_edge_usd) - float(intent.expected_cost_usd),
                order_risk_pct=float(intent.estimated_risk_usd) / max(float(state.equity_usd), 1e-12),
                total_open_risk_pct_after=(
                    float(state.open_risk_usd) + float(intent.estimated_risk_usd)
                ) / max(float(state.equity_usd), 1e-12),
            )
            return ExecutionResult(False, duplicate, None)

        decision = self.safeguard.evaluate(intent, state)
        if not decision.approved:
            return ExecutionResult(False, decision, None)

        self._seen_decision_ids.add(intent.decision_id)
        fill = PaperFill(
            decision_id=intent.decision_id,
            symbol=intent.symbol,
            notional_usd=float(intent.notional_usd),
            estimated_risk_usd=float(intent.estimated_risk_usd),
        )
        return ExecutionResult(True, decision, fill)
