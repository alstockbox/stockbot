import pytest
from pydantic import ValidationError

from stockbot.ai.researcher import ResearchScientist
from stockbot.llm.schemas import ResearchHypothesis


class BadProvider:
    def generate_json(self, context):
        return {"hypotheses": [{"hypothesis": "Buy SPY now", "evidence": ["up"], "proposed_change": "place live order BUY SPY", "validation_plan": ["none"], "invalidation_criteria": ["none"]}]}


def test_hypothesis_schema_rejects_direct_live_order_instructions():
    with pytest.raises(ValidationError):
        ResearchHypothesis(
            hypothesis="Immediate trade",
            evidence=["price moved"],
            proposed_change="execute live order BUY SPY now",
            validation_plan=["none"],
            invalidation_criteria=["none"],
        )


def test_provider_output_cannot_bypass_structured_safety_contract():
    with pytest.raises(ValidationError):
        ResearchScientist(provider=BadProvider()).review({"metrics": {}})


def test_fallback_researcher_generates_testable_cost_hypothesis():
    ideas = ResearchScientist().review({"metrics": {"cost_ratio": 0.25, "max_drawdown": 0.05}, "regime_degradation": False})
    assert ideas
    assert any("turnover" in idea.proposed_change.lower() or "cost" in idea.hypothesis.lower() for idea in ideas)
