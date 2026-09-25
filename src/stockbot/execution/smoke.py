from __future__ import annotations

from dataclasses import dataclass

from stockbot.execution.paper import PaperExecutor
from stockbot.execution.safeguard import (
    ExecutionIntent,
    ExecutionMode,
    SafeguardGate,
    SafeguardState,
)


@dataclass(frozen=True)
class SmokeCase:
    name: str
    intent: ExecutionIntent
    state: SafeguardState
    should_pass: bool
    expected_reason: str | None = None


def _intent(
    decision_id: str,
    *,
    mode: ExecutionMode = ExecutionMode.PAPER,
    risk: float = 1.0,
    notional: float = 100.0,
    gross_edge: float = 1.0,
    cost: float = 0.10,
) -> ExecutionIntent:
    return ExecutionIntent(
        decision_id,
        "EURUSD",
        mode,
        risk,
        notional,
        gross_edge,
        cost,
    )


def default_smoke_cases(equity_usd: float = 466.0) -> list[SmokeCase]:
    safe = SafeguardState(
        equity_usd=equity_usd,
        equity_peak_usd=equity_usd,
        data_age_seconds=1.0,
    )
    return [
        SmokeCase(
            "small_positive_edge_paper",
            _intent("ok-1", risk=1.50, notional=150.0),
            safe,
            True,
        ),
        SmokeCase(
            "oversized_order_risk",
            _intent("risk-1", risk=5.00),
            safe,
            False,
            "ORDER_RISK_LIMIT",
        ),
        SmokeCase(
            "total_open_risk_limit",
            _intent("total-risk-1", risk=2.0),
            SafeguardState(
                equity_usd,
                equity_usd,
                open_risk_usd=8.0,
                data_age_seconds=1.0,
            ),
            False,
            "TOTAL_OPEN_RISK_LIMIT",
        ),
        SmokeCase(
            "stale_market_data",
            _intent("stale-1"),
            SafeguardState(equity_usd, equity_usd, data_age_seconds=120.0),
            False,
            "STALE_DATA",
        ),
        SmokeCase(
            "daily_loss_circuit_breaker",
            _intent("loss-1"),
            SafeguardState(equity_usd, equity_usd, daily_pnl_usd=-10.0),
            False,
            "DAILY_LOSS_LIMIT",
        ),
        SmokeCase(
            "drawdown_circuit_breaker",
            _intent("dd-1"),
            SafeguardState(
                equity_usd=400.0,
                equity_peak_usd=466.0,
                data_age_seconds=1.0,
            ),
            False,
            "MAX_DRAWDOWN",
        ),
        SmokeCase(
            "kill_switch",
            _intent("kill-1"),
            SafeguardState(
                equity_usd,
                equity_usd,
                data_age_seconds=1.0,
                kill_switch=True,
            ),
            False,
            "KILL_SWITCH",
        ),
        SmokeCase(
            "gross_exposure_limit",
            _intent("gross-1", notional=200.0),
            SafeguardState(
                equity_usd,
                equity_usd,
                gross_notional_usd=800.0,
                data_age_seconds=1.0,
            ),
            False,
            "GROSS_EXPOSURE_LIMIT",
        ),
        SmokeCase(
            "order_rate_limit",
            _intent("rate-1"),
            SafeguardState(
                equity_usd,
                equity_usd,
                data_age_seconds=1.0,
                orders_last_minute=10,
            ),
            False,
            "ORDER_RATE_LIMIT",
        ),
        SmokeCase(
            "no_net_edge",
            _intent("edge-1", gross_edge=0.05, cost=0.10),
            safe,
            False,
            "NO_POSITIVE_NET_EDGE",
        ),
        SmokeCase(
            "cost_consumes_edge",
            _intent("cost-1", gross_edge=0.10, cost=0.10),
            safe,
            False,
            "COST_CONSUMES_EDGE",
        ),
        SmokeCase(
            "live_is_fail_closed",
            _intent("live-1", mode=ExecutionMode.LIVE),
            safe,
            False,
            "LIVE_EXECUTION_DISABLED",
        ),
    ]


def run_smoke_test(equity_usd: float = 466.0) -> dict[str, bool]:
    executor = PaperExecutor(SafeguardGate())
    results: dict[str, bool] = {}
    for case in default_smoke_cases(equity_usd):
        result = executor.submit(case.intent, case.state)
        ok = result.accepted is case.should_pass
        if case.expected_reason is not None:
            ok = ok and case.expected_reason in result.safeguard.reasons
        results[case.name] = ok
    return results
