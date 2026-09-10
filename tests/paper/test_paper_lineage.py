from dataclasses import replace
from types import SimpleNamespace

import pytest

from stockbot.data.schemas import DataGrade
from stockbot.paper.arena import PaperArenaCriteria, evaluate_paper_track
from stockbot.paper.deployment_gate import evaluate_deployment_evidence
from stockbot.paper.ledger import PaperObservation


def _row(
    day: int,
    *,
    artifact_id: str | None = "artifact-a",
    cycle_id: str | None = "cycle-a",
    signal_snapshot: str | None = "signal-snapshot",
    realization_snapshot: str | None = "realization-snapshot",
    signal_timestamp: str | None = None,
) -> PaperObservation:
    realization = f"2026-09-{day:02d}T16:00:00+00:00"
    signal = signal_timestamp or f"2026-09-{day:02d}T15:00:00+00:00"
    return PaperObservation(
        strategy_id="strategy-lineage",
        timestamp=realization,
        net_return=0.001,
        benchmark_return=0.0,
        turnover=0.2,
        cost_rate=0.0001,
        fill_rate=0.99,
        signal_count=3,
        research_cycle_id=cycle_id,
        model_artifact_id=artifact_id,
        signal_timestamp=signal,
        signal_snapshot_fingerprint=signal_snapshot,
        realization_snapshot_fingerprint=realization_snapshot,
    )


def _lenient_require_provenance() -> PaperArenaCriteria:
    return PaperArenaCriteria(
        min_sessions=1,
        min_live_span_days=1,
        min_sharpe=-100.0,
        min_excess_cagr=-100.0,
        max_drawdown=1.0,
        min_fill_rate=0.0,
        max_average_turnover=100.0,
        min_bootstrap_confidence=0.0,
        reject_drift=False,
        require_frozen_provenance=True,
    )


def test_paper_track_rejects_mixed_model_artifact_lineage():
    rows = [_row(1), _row(2, artifact_id="artifact-b")]
    with pytest.raises(ValueError, match="model artifact"):
        evaluate_paper_track(rows)


def test_paper_track_rejects_mixed_or_partial_research_cycle_lineage():
    with pytest.raises(ValueError, match="research cycle"):
        evaluate_paper_track([_row(1), _row(2, cycle_id="cycle-b")])

    with pytest.raises(ValueError, match="research cycle"):
        evaluate_paper_track([_row(1), _row(2, cycle_id=None)])


def test_missing_frozen_provenance_blocks_deployment_grade_paper_track():
    row = replace(
        _row(1),
        model_artifact_id=None,
        research_cycle_id=None,
        signal_timestamp=None,
        signal_snapshot_fingerprint=None,
        realization_snapshot_fingerprint=None,
    )
    report = evaluate_paper_track([row], criteria=_lenient_require_provenance())

    assert not report.frozen_provenance_complete
    assert not report.live_eligible
    assert "paper_frozen_provenance" in report.reasons


def test_complete_single_frozen_lineage_is_reported_explicitly():
    rows = [_row(1), _row(2)]
    report = evaluate_paper_track(rows, criteria=_lenient_require_provenance())

    assert report.frozen_provenance_complete
    assert report.model_artifact_id == "artifact-a"
    assert report.research_cycle_id == "cycle-a"
    assert "paper_frozen_provenance" not in report.reasons


def test_deployment_gate_rejects_live_eligible_paper_without_frozen_provenance():
    audit = SimpleNamespace(holdout_report=SimpleNamespace(passed=True))
    paper = SimpleNamespace(live_eligible=True, frozen_provenance_complete=False)

    report = evaluate_deployment_evidence(
        research_ready=True,
        data_grade=DataGrade.RESEARCH_GRADE,
        quarantine_audit=audit,
        paper_report=paper,
    )

    assert not report.paper_frozen_provenance_complete
    assert not report.eligible_for_manual_live_review
    assert "forward_paper_provenance_not_verified" in report.reasons
