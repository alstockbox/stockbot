from __future__ import annotations

from dataclasses import dataclass

from stockbot.data.schemas import DataGrade
from stockbot.paper.arena import PaperArenaReport
from stockbot.research.quarantine_audit import QuarantineAuditResult


@dataclass(frozen=True)
class DeploymentEvidenceReport:
    research_ready: bool
    research_grade: bool
    quarantine_audit_passed: bool
    paper_live_evidence_eligible: bool
    paper_frozen_provenance_complete: bool
    paper_external_provenance_verified: bool
    hard_risk_engine_required: bool
    broker_execution_available: bool
    eligible_for_manual_live_review: bool
    reasons: tuple[str, ...]


def _external_paper_provenance_complete(report: object | None) -> bool:
    if report is None or not bool(getattr(report, "verified", False)):
        return False
    boolean_evidence = (
        "artifact_verified",
        "model_hash_verified",
        "artifact_strategy_match",
        "artifact_cycle_match",
        "snapshot_timeline_verified",
    )
    if not all(bool(getattr(report, field, False)) for field in boolean_evidence):
        return False
    count_evidence = (
        "signal_snapshots_verified",
        "realization_snapshots_verified",
        "observations_verified",
    )
    try:
        if not all(int(getattr(report, field, 0)) > 0 for field in count_evidence):
            return False
    except (TypeError, ValueError):
        return False
    provenance_reasons = getattr(report, "reasons", None)
    return provenance_reasons is not None and not tuple(provenance_reasons)


def evaluate_deployment_evidence(
    *,
    research_ready: bool,
    data_grade: DataGrade,
    quarantine_audit: QuarantineAuditResult | None,
    paper_report: PaperArenaReport | None,
    paper_provenance_report: object | None = None,
) -> DeploymentEvidenceReport:
    """Combine independent evidence layers without granting execution authority.

    Forward-paper performance is not deployment evidence unless every session is bound
    to one frozen model artifact/research cycle with complete signal and realization
    provenance *and* a separate verifier has confirmed the artifact identity/model hash,
    strategy/cycle bindings, immutable signal/realization snapshots and timeline. A
    high-performing manual, mixed-refit, partial or merely self-declared track therefore
    cannot satisfy the manual-live-review gate.
    """

    research_grade = data_grade is DataGrade.RESEARCH_GRADE
    audit_passed = quarantine_audit is not None and quarantine_audit.holdout_report.passed
    paper_eligible = paper_report is not None and bool(paper_report.live_eligible)
    paper_provenance = paper_report is not None and bool(
        getattr(paper_report, "frozen_provenance_complete", False)
    )
    paper_external_provenance = _external_paper_provenance_complete(
        paper_provenance_report
    )

    reasons: list[str] = []
    if not research_ready:
        reasons.append("research_not_ready")
    if not research_grade:
        reasons.append("data_not_research_grade")
    if not audit_passed:
        reasons.append("sealed_quarantine_audit_not_passed")
    if not paper_eligible:
        reasons.append("forward_paper_evidence_not_ready")
    if not paper_provenance:
        reasons.append("forward_paper_provenance_not_verified")
    if not paper_external_provenance:
        reasons.append("forward_paper_external_provenance_not_verified")

    eligible = not reasons
    return DeploymentEvidenceReport(
        research_ready=bool(research_ready),
        research_grade=research_grade,
        quarantine_audit_passed=audit_passed,
        paper_live_evidence_eligible=paper_eligible,
        paper_frozen_provenance_complete=paper_provenance,
        paper_external_provenance_verified=paper_external_provenance,
        hard_risk_engine_required=True,
        broker_execution_available=False,
        eligible_for_manual_live_review=eligible,
        reasons=tuple(reasons),
    )
