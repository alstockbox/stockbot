from stockbot.domain.models import TargetPosition
from stockbot.risk.engine import RiskConfig, RiskEngine, RiskState


def test_risk_engine_caps_position_without_bypass():
    engine = RiskEngine(RiskConfig(max_position_weight=0.25))
    decision = engine.evaluate(TargetPosition("SPY", 0.8), RiskState())
    assert decision.approved
    assert decision.target.weight == 0.25
    assert "POSITION_CAPPED" in decision.reasons


def test_risk_engine_rejects_stale_data():
    engine = RiskEngine(RiskConfig(max_data_age_seconds=60))
    decision = engine.evaluate(TargetPosition("SPY", 0.2), RiskState(data_age_seconds=61))
    assert not decision.approved
    assert "STALE_DATA" in decision.reasons


def test_risk_engine_rejects_after_max_drawdown():
    engine = RiskEngine(RiskConfig(max_drawdown=0.10))
    state = RiskState(equity=89, equity_peak=100)
    decision = engine.evaluate(TargetPosition("SPY", 0.2), state)
    assert not decision.approved
    assert "MAX_DRAWDOWN" in decision.reasons
