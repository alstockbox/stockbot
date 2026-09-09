import math

from stockbot.research.readiness import evaluate_paper_readiness


def test_paper_readiness_requires_complete_evidence_chain():
    report = evaluate_paper_readiness(
        oos_coverage=0.80,
        robustness=0.75,
        stress_score=0.70,
        regime_score=0.60,
        discovery_q_value=0.05,
        drift_score=0.80,
        drift_degraded=False,
        holdout_passed=True,
        holdout_score=1.0,
    )
    assert report.ready
    assert report.reasons == ()
    assert 0.0 <= report.score <= 1.0


def test_paper_readiness_rejects_statistically_weak_or_drifting_candidate():
    report = evaluate_paper_readiness(
        oos_coverage=0.80,
        robustness=0.75,
        stress_score=0.70,
        regime_score=0.60,
        discovery_q_value=0.80,
        drift_score=0.20,
        drift_degraded=True,
        holdout_passed=True,
        holdout_score=0.5,
    )
    assert not report.ready
    assert "false_discovery_risk" in report.reasons
    assert "recent_drift" in report.reasons


def test_paper_readiness_fails_closed_on_non_finite_evidence():
    report = evaluate_paper_readiness(
        oos_coverage=float("nan"),
        robustness=0.75,
        stress_score=0.70,
        regime_score=0.60,
        discovery_q_value=0.05,
        drift_score=0.80,
        drift_degraded=False,
        holdout_passed=True,
        holdout_score=1.0,
    )

    assert not report.ready
    assert "non_finite_evidence" in report.reasons
    assert math.isfinite(report.score)
