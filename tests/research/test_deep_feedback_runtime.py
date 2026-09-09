from __future__ import annotations

from types import SimpleNamespace

import pytest

from stockbot.research.deep_feedback import DeepResearchFinding, JsonlDeepResearchMemory
from stockbot.research.market_training import run_snapshot_deep_diagnostics, run_snapshot_factory
from stockbot.research.memory import ExperimentRecord, JsonlExperimentMemory


def _record(experiment_id: str, score: float) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=experiment_id,
        model_name="ridge",
        model_params={"alpha": 1.0},
        seed=7,
        dataset_fingerprint=f"dataset-{experiment_id}",
        horizon=5,
        factory_score=score,
        base_score=score,
        robustness=0.8,
        oos_coverage=0.9,
        metrics={"sharpe": 1.0},
        passed_gates=True,
        rejection_reasons=(),
        created_at="2026-09-01T00:00:00+00:00",
        stress_score=0.8,
    )


def _finding(experiment_id: str, *, cycle: str, priority: float) -> DeepResearchFinding:
    return DeepResearchFinding(
        finding_id=f"finding-{cycle}-{experiment_id}",
        source_cycle_id=cycle,
        experiment_id=experiment_id,
        horizon=5,
        model_name="ridge",
        created_at="2026-09-01T00:00:00+00:00",
        adaptive_priority=priority,
        bootstrap_confidence=priority,
        factor_idiosyncratic_score=priority,
        factor_explained_fraction=0.2,
        capacity_score=priority,
        max_effective_capital=100_000.0,
        window_robustness_score=priority,
        useful_feature_groups=(),
        harmful_feature_groups=(),
        best_top_fraction=0.2,
        best_weighting="equal",
        neutralization_score_delta=None,
        sector_cap_score_delta=None,
        sector_concentration_reduction=None,
        stacking_score=None,
        stacking_delta_vs_policy=None,
    )


def _deep_diagnostics():
    candidate = SimpleNamespace(
        experiment_id="exp-a",
        horizon=5,
        model_name="ridge",
        policy_arena=SimpleNamespace(
            best=SimpleNamespace(
                policy=SimpleNamespace(top_fraction=0.2, weighting="equal"),
                score=1.0,
            )
        ),
        bootstrap_uncertainty=SimpleNamespace(confidence_score=0.8),
        factor_exposure=SimpleNamespace(
            idiosyncratic_score=0.7,
            explained_fraction=0.2,
        ),
        capacity_curve=SimpleNamespace(
            capacity_score=0.6,
            max_effective_capital=250_000.0,
        ),
        window_robustness=SimpleNamespace(score=0.9),
        feature_ablation=SimpleNamespace(results=(), recommended_drop_groups=()),
        neutralization=None,
        sector_cap=None,
    )
    return SimpleNamespace(candidates={"exp-a": candidate}, stacking_reports={})


def test_snapshot_factory_uses_prior_cycle_deep_feedback_but_not_same_cycle(tmp_path, monkeypatch):
    experiment_path = tmp_path / "experiments.jsonl"
    experiment_memory = JsonlExperimentMemory(experiment_path)
    experiment_memory.append(_record("exp-a", 1.00))
    experiment_memory.append(_record("exp-b", 1.01))

    deep_path = tmp_path / "deep-findings.jsonl"
    deep_memory = JsonlDeepResearchMemory(deep_path)
    deep_memory.append(_finding("exp-a", cycle="cycle-old", priority=1.0))

    captured = {}

    class FakeFactory:
        def __init__(self, config, *, memory, champion_store):
            captured["config"] = config
            captured["memory"] = memory
            captured["champion_store"] = champion_store

        def run(self, bars, metadata, **kwargs):
            captured["bars"] = bars
            captured["metadata"] = metadata
            return "factory-report"

    monkeypatch.setattr("stockbot.research.market_training.ResearchFactory", FakeFactory)
    monkeypatch.setattr("stockbot.research.market_training._development_bars", lambda *args, **kwargs: "BARS")
    monkeypatch.setattr("stockbot.research.market_training._snapshot_metadata", lambda *args, **kwargs: "META")

    result = run_snapshot_factory(
        object(),
        memory_path=str(experiment_path),
        deep_feedback_path=deep_path,
        research_cycle_id="cycle-new",
        deep_feedback_weight=0.30,
    )

    assert result == "factory-report"
    parents = captured["config"].population.adaptive_records
    assert [row.experiment_id for row in parents[:2]] == ["exp-a", "exp-b"]

    captured.clear()
    run_snapshot_factory(
        object(),
        memory_path=str(experiment_path),
        deep_feedback_path=deep_path,
        research_cycle_id="cycle-old",
        deep_feedback_weight=0.30,
    )
    parents = captured["config"].population.adaptive_records
    assert [row.experiment_id for row in parents[:2]] == ["exp-b", "exp-a"]


def test_snapshot_deep_diagnostics_appends_current_cycle_findings(tmp_path, monkeypatch):
    diagnostics = _deep_diagnostics()
    deep_path = tmp_path / "deep-findings.jsonl"

    monkeypatch.setattr("stockbot.research.market_training._development_bars", lambda *args, **kwargs: "DEVELOPMENT")
    monkeypatch.setattr("stockbot.research.market_training._snapshot_metadata", lambda *args, **kwargs: "META")

    def fake_deep(bars, metadata, report, **kwargs):
        assert bars == "DEVELOPMENT"
        assert metadata == "META"
        return diagnostics

    monkeypatch.setattr("stockbot.research.market_training.run_deep_research_diagnostics", fake_deep)

    returned = run_snapshot_deep_diagnostics(
        object(),
        object(),
        deep_feedback_path=deep_path,
        research_cycle_id="cycle-current",
    )

    assert returned is diagnostics
    stored = JsonlDeepResearchMemory(deep_path).records()
    assert len(stored) == 1
    assert stored[0].experiment_id == "exp-a"
    assert stored[0].source_cycle_id == "cycle-current"


def test_snapshot_feedback_runtime_requires_path_and_cycle_together(tmp_path):
    with pytest.raises(ValueError, match="deep feedback"):
        run_snapshot_factory(
            object(),
            deep_feedback_path=tmp_path / "deep-findings.jsonl",
        )

    with pytest.raises(ValueError, match="deep feedback"):
        run_snapshot_deep_diagnostics(
            object(),
            object(),
            research_cycle_id="cycle-only",
        )
