from stockbot.arena.experiments import ModelExperimentResult
from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.ml.artifacts import ExperimentArtifact
from stockbot.research.gates import evaluate_research_gate
from stockbot.research.objective import risk_adjusted_objective


def _result(metrics: dict[str, float], robustness: float = 0.8, oos_coverage: float = 0.8):
    artifact = ExperimentArtifact(
        dataset_fingerprint="abc",
        feature_names=("x",),
        label_name="target",
        model_name="ridge",
        model_params={"alpha": 1.0},
        seed=7,
        fold_count=4,
        oos_coverage=oos_coverage,
        metrics=metrics,
    )
    return ModelExperimentResult(
        name="ridge",
        score=1.0,
        metrics=metrics,
        robustness=robustness,
        oos_coverage=oos_coverage,
        artifact=artifact,
    )


def test_objective_penalizes_fragile_high_risk_curve():
    stable = {
        "cagr": 0.18,
        "sharpe": 1.4,
        "sortino": 1.8,
        "calmar": 1.2,
        "max_drawdown": 0.12,
        "cvar_95": 0.015,
        "volatility": 0.16,
        "turnover": 8.0,
        "instability": 0.15,
        "concentration": 0.20,
    }
    fragile = dict(stable)
    fragile.update({"cagr": 0.30, "max_drawdown": 0.45, "cvar_95": 0.08, "turnover": 90.0})

    stable_score = risk_adjusted_objective(stable, robustness=0.85, oos_coverage=0.8)
    fragile_score = risk_adjusted_objective(fragile, robustness=0.45, oos_coverage=0.8)

    assert stable_score > fragile_score


def test_gate_requires_research_grade_and_rejects_excess_drawdown():
    metrics = {
        "max_drawdown": 0.35,
        "cvar_95": 0.02,
        "turnover": 5.0,
        "instability": 0.1,
        "concentration": 0.2,
    }
    result = _result(metrics)
    bootstrap = DatasetMetadata("sample", "provider", DataGrade.BOOTSTRAP)
    decision = evaluate_research_gate(result, bootstrap, factory_score=1.0)

    assert not decision.passed
    assert "dataset_not_research_grade" in decision.reasons
    assert "drawdown_limit" in decision.reasons


def test_gate_allows_robust_research_grade_candidate():
    metrics = {
        "max_drawdown": 0.12,
        "cvar_95": 0.015,
        "turnover": 6.0,
        "instability": 0.1,
        "concentration": 0.2,
    }
    result = _result(metrics)
    metadata = DatasetMetadata("sample", "provider", DataGrade.RESEARCH_GRADE)
    decision = evaluate_research_gate(result, metadata, factory_score=1.0)

    assert decision.passed
    assert decision.reasons == ()
