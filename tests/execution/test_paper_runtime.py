from datetime import datetime, timezone

from stockbot.execution.safeguard import SafeguardState
from stockbot.opportunity.ranker import Opportunity, OpportunityPolicy
from stockbot.runtime.paper_runtime import PaperTradePlan, PaperTradingRuntime


def test_paper_runtime_ranks_executes_and_journals(tmp_path):
    runtime = PaperTradingRuntime(
        tmp_path / "journal.jsonl",
        opportunity_policy=OpportunityPolicy(minimum_net_edge=0.0005),
    )
    plans = [
        PaperTradePlan(
            "a",
            Opportunity("AAA", "trend", 0.010, 0.001, 0.002, 0.001, 0.001, 0.9),
            0.003,
            0.2,
        ),
        PaperTradePlan(
            "b",
            Opportunity("BBB", "weak", 0.002, 0.001, 0.003, 0.001, 0.001, 0.6),
            0.003,
            0.2,
        ),
    ]
    result = runtime.run_cycle(
        plans,
        SafeguardState(466.0, 466.0, data_age_seconds=1.0),
        timestamp=datetime(2026, 9, 26, tzinfo=timezone.utc),
        regime="neutral_chop",
    )
    assert len(result.ranked) == 1
    assert result.accepted == 1
    assert len(runtime.journal.entries()) == 1


def test_runtime_accumulates_open_risk_within_cycle(tmp_path):
    runtime = PaperTradingRuntime(tmp_path / "journal.jsonl")
    plans = [
        PaperTradePlan(
            str(index),
            Opportunity(
                f"S{index}",
                f"model-{index}",
                0.02,
                0.001,
                0.001,
                0.0,
                0.0,
                0.9,
            ),
            0.005,
            0.2,
        )
        for index in range(5)
    ]
    result = runtime.run_cycle(
        plans,
        SafeguardState(466.0, 466.0, data_age_seconds=1.0),
        timestamp=datetime(2026, 9, 26, tzinfo=timezone.utc),
        regime="bull_trend",
    )
    assert result.accepted == 4
    assert result.rejected == 1
    assert "TOTAL_OPEN_RISK_LIMIT" in result.executions[-1].safeguard.reasons
