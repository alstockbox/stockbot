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
    hard_risk_engine_required: bool
    broker_execution_available: bool
    eligible_for_manual_live_review: bool
    reasons: tuple[str, ...]


def evaluate_deployment_evidence(
    *,
    research_ready: bool,
    data_grade: DataGrade,
    quarantine_audit: QuarantineAuditResult | None,
    paper_report: PaperArenaReport | None,
) -> DeploymentEvidenceReport:
    """Combine independent evidence layers without granting execution authority.

    Forward-paper performance is not deployment evidence unless every session is bound
    to one frozen model artifact/research cycle with complete signal and realization
    provenance. A high-performing manual or mixed-refit track therefore cannot satisfy
    the manual-live-review gate.
    """

    research_grade = data_grade is DataGrade.RESEARCH_GRADE
    audit_passed = quarantine_audit is not None and quarantine_audit.holdout_report.passed
    paper_eligible = paper_report is not None and bool(paper_report.live_eligible)
    paper_provenance = paper_report is not None and bool(
        getattr(paper_report, "frozen_provenance_complete", False)
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

    eligible = not reasons
    return DeploymentEvidenceReport(
        research_ready=bool(research_ready),
        research_grade=research_grade,
        quarantine_audit_passed=audit_passed,
        paper_live_evidence_eligible=paper_eligible,
        paper_frozen_provenance_complete=paper_provenance,
        hard_risk_engine_required=True,
        broker_execution_available=False,
        eligible_for_manual_live_review=eligible,
        reasons=tuple(reasons),
    )
