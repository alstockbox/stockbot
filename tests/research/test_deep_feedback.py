from __future__ import annotations

from types import SimpleNamespace

from stockbot.research.deep_feedback import (
    JsonlDeepResearchMemory,
    build_research_cycle_id,
    make_deep_findings,
    rank_adaptive_parent_records,
)
from stockbot.research.memory import ExperimentRecord


def _record(experiment_id: str, factory_score: float) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=experiment_id,
        model_name="ridge",
        model_params={"alpha": 1.0},
        seed=7,
        dataset_fingerprint=f"dataset-{experiment_id}",
        horizon=5,
        factory_score=factory_score,
        base_score=factory_score,
        robustness=0.8,
        oos_coverage=0.9,
        metrics={"sharpe": 1.0},
        passed_gates=True,
        rejection_reasons=(),
        created_at="2026-09-01T00:00:00+00:00",
        stress_score=0.8,
    )


def _diagnostics():
    feature_results = (
        SimpleNamespace(group="momentum_returns", score_impact=0.20),
        SimpleNamespace(group="volatility", score_impact=-0.12),
        SimpleNamespace(group="liquidity_volume", score_impact=0.03),
    )
    candidate = SimpleNamespace(
        experiment_id="exp-a",
        horizon=5,
        model_name="ridge",
        policy_arena=SimpleNamespace(
            best=SimpleNamespace(
                policy=SimpleNamespace(top_fraction=0.20, weighting="conviction"),
                score=1.30,
            )
        ),
        bootstrap_uncertainty=SimpleNamespace(confidence_score=0.80),
        factor_exposure=SimpleNamespace(
            idiosyncratic_score=0.70,
            explained_fraction=0.25,
        ),
        capacity_curve=SimpleNamespace(
            capacity_score=0.60,
            max_effective_capital=500_000.0,
        ),
        window_robustness=SimpleNamespace(score=0.90),
        feature_ablation=SimpleNamespace(
            results=feature_results,
            recommended_drop_groups=("volatility",),
        ),
        neutralization=SimpleNamespace(score_delta=0.05),
        sector_cap=SimpleNamespace(
            score_delta=0.08,
            baseline_sector_exposure=SimpleNamespace(average_max_sector_weight=0.55),
            constrained_sector_exposure=SimpleNamespace(average_max_sector_weight=0.32),
        ),
    )
    stacking = SimpleNamespace(score=1.45)
    return SimpleNamespace(
        candidates={"exp-a": candidate},
        stacking_reports={5: stacking},
    )


def test_research_cycle_id_is_data_identity_not_execution_knobs():
    first = build_research_cycle_id(
        dataset_fingerprint="bars-123",
        quarantine_start="2026-08-01",
        universe_fingerprint="universe-1",
        auxiliary_fingerprint="aux-1",
        quality_fingerprint="quality-1",
    )
    same = build_research_cycle_id(
        dataset_fingerprint="bars-123",
        quarantine_start="2026-08-01",
        universe_fingerprint="universe-1",
        auxiliary_fingerprint="aux-1",
        quality_fingerprint="quality-1",
    )
    changed_data = build_research_cycle_id(
        dataset_fingerprint="bars-124",
        quarantine_start="2026-08-01",
        universe_fingerprint="universe-1",
        auxiliary_fingerprint="aux-1",
        quality_fingerprint="quality-1",
    )

    assert first == same
    assert first != changed_data
    assert len(first) == 24


def test_make_deep_findings_compacts_only_development_diagnostics():
    findings = make_deep_findings(
        _diagnostics(),
        source_cycle_id="cycle-old",
        useful_feature_threshold=0.05,
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.experiment_id == "exp-a"
    assert finding.source_cycle_id == "cycle-old"
    assert finding.best_top_fraction == 0.20
    assert finding.best_weighting == "conviction"
    assert finding.bootstrap_confidence == 0.80
    assert finding.factor_idiosyncratic_score == 0.70
    assert finding.factor_explained_fraction == 0.25
    assert finding.capacity_score == 0.60
    assert finding.max_effective_capital == 500_000.0
    assert finding.window_robustness_score == 0.90
    assert finding.useful_feature_groups == ("momentum_returns",)
    assert finding.harmful_feature_groups == ("volatility",)
    assert finding.neutralization_score_delta == 0.05
    assert finding.sector_cap_score_delta == 0.08
    assert finding.sector_concentration_reduction == 0.23
    assert finding.stacking_score == 1.45
    assert finding.stacking_delta_vs_policy == 0.15
    assert 0.0 <= finding.adaptive_priority <= 1.0


def test_deep_memory_is_append_only_deduplicated_and_excludes_current_cycle(tmp_path):
    memory = JsonlDeepResearchMemory(tmp_path / "deep-findings.jsonl")
    finding = make_deep_findings(_diagnostics(), source_cycle_id="cycle-old")[0]
    memory.append(finding)
    memory.append(finding)

    assert len(memory.records()) == 1
    assert memory.for_future_cycle("cycle-new") == [finding]
    assert memory.for_future_cycle("cycle-old") == []


def test_parent_ranking_uses_only_prior_cycle_feedback():
    records = [_record("exp-a", 1.00), _record("exp-b", 1.01)]
    strong = make_deep_findings(_diagnostics(), source_cycle_id="cycle-old")[0]

    prior_ranked = rank_adaptive_parent_records(
        records,
        [strong],
        current_cycle_id="cycle-new",
        limit=2,
        feedback_weight=0.30,
    )
    same_cycle_ranked = rank_adaptive_parent_records(
        records,
        [strong],
        current_cycle_id="cycle-old",
        limit=2,
        feedback_weight=0.30,
    )

    assert prior_ranked[0].experiment_id == "exp-a"
    assert same_cycle_ranked[0].experiment_id == "exp-b"
