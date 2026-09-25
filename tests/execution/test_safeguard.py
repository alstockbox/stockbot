import pytest

from stockbot.execution.paper import PaperExecutor
from stockbot.execution.safeguard import (
    ExecutionIntent,
    ExecutionMode,
    SafeguardGate,
    SafeguardState,
)
from stockbot.execution.smoke import run_smoke_test


def safe_state():
    return SafeguardState(
        equity_usd=466.0,
        equity_peak_usd=466.0,
        data_age_seconds=1.0,
    )


def test_safeguard_smoke_matrix_all_passes():
    results = run_smoke_test(466.0)
    assert results
    assert all(results.values())


def test_duplicate_decision_is_rejected():
    executor = PaperExecutor(SafeguardGate())
    intent = ExecutionIntent(
        "same-id",
        "EURUSD",
        ExecutionMode.PAPER,
        1.0,
        100.0,
        1.0,
        0.1,
    )
    assert executor.submit(intent, safe_state()).accepted
    duplicate = executor.submit(intent, safe_state())
    assert not duplicate.accepted
    assert duplicate.safeguard.reasons == ("DUPLICATE_DECISION_ID",)


def test_live_execution_is_disabled_by_default():
    executor = PaperExecutor(SafeguardGate())
    intent = ExecutionIntent(
        "live",
        "EURUSD",
        ExecutionMode.LIVE,
        1.0,
        100.0,
        1.0,
        0.1,
    )
    result = executor.submit(intent, safe_state())
    assert not result.accepted
    assert "LIVE_EXECUTION_DISABLED" in result.safeguard.reasons


def test_default_466_account_caps_order_risk_near_half_percent():
    gate = SafeguardGate()
    approved = gate.evaluate(
        ExecutionIntent("a", "EURUSD", ExecutionMode.PAPER, 2.0, 100.0, 1.0, 0.1),
        safe_state(),
    )
    rejected = gate.evaluate(
        ExecutionIntent("b", "EURUSD", ExecutionMode.PAPER, 2.5, 100.0, 1.0, 0.1),
        safe_state(),
    )
    assert approved.approved
    assert not rejected.approved
    assert "ORDER_RISK_LIMIT" in rejected.reasons
