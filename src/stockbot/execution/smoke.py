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


def default_smoke_cases(equity_usd: float = 466.0) -> list[SmokeCase]:
    safe = SafeguardState(
        equity_usd=equity_usd,
        equity_peak_usd=equity_usd,
        data_age_seconds=1.0,
    )
    return [
        SmokeCase(
            "small_positive_edge_paper",
            ExecutionIntent("ok-1", "EURUSD", ExecutionMode.PAPER, 1.50, 150.0, 1.00, 0.10),
            safe,
            True,
        ),
        SmokeCase(
            "oversized_order_risk",
            ExecutionIntent("risk-1", "EURUSD", ExecutionMode.PAPER, 5.00, 150.0, 1.00, 0.10),
            safe,
            False,
            "ORDER_RISK_LIMIT",
        ),
        SmokeCase(
            "stale_market_data",
            ExecutionIntent("stale-1", "EURUSD", ExecutionMode.PAPER, 1.00, 100.0, 1.00, 0.10),
            SafeguardState(equity_usd, equity_usd, data_age_seconds=120.0),
            False,
            "STALE_DATA",
        ),
        SmokeCase(
            "daily_loss_circuit_breaker",
            ExecutionIntent("loss-1", "EURUSD", ExecutionMode.PAPER, 1.00, 100.0, 1.00, 0.10),
            SafeguardState(equity_usd, equity_usd, daily_pnl_usd=-10.0),
            False,
            "DAILY_LOSS_LIMIT",
        ),
        SmokeCase(
            "no_net_edge",
            ExecutionIntent("edge-1", "EURUSD", ExecutionMode.PAPER, 1.00, 100.0, 0.05, 0.10),
            safe,
            False,
            "NO_POSITIVE_NET_EDGE",
        ),
        SmokeCase(
            "live_is_fail_closed",
            ExecutionIntent("live-1", "EURUSD", ExecutionMode.LIVE, 1.00, 100.0, 1.00, 0.10),
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
