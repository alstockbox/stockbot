from datetime import datetime, timezone

import pytest

from stockbot.domain.models import RiskDecision, Signal, TargetPosition


def test_signal_score_must_be_bounded():
    with pytest.raises(ValueError):
        Signal(symbol="SPY", score=1.01, source="test", timestamp=datetime.now(timezone.utc))


def test_target_weight_must_be_long_only_and_unlevered():
    with pytest.raises(ValueError):
        TargetPosition(symbol="SPY", weight=-0.01)
    with pytest.raises(ValueError):
        TargetPosition(symbol="SPY", weight=1.01)


def test_risk_decision_preserves_machine_readable_reason_codes():
    target = TargetPosition(symbol="SPY", weight=0.25)
    decision = RiskDecision(approved=False, target=target, reasons=("MAX_DRAWDOWN",))
    assert decision.reasons == ("MAX_DRAWDOWN",)
