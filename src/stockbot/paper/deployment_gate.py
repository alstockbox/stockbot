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
    """Combine independent evidence layers without granting execution authority."""

    research_grade = data_grade is DataGrade.RESEARCH_GRADE
    audit_passed = quarantine_audit is not None and quarantine_audit.holdout_report.passed
    paper_eligible = paper_report is not None and paper_report.live_eligible

    reasons: list[str] = []
    if not research_ready:
        reasons.append("research_not_ready")
    if not research_grade:
        reasons.append("data_not_research_grade")
    if not audit_passed:
        reasons.append("sealed_quarantine_audit_not_passed")
    if not paper_eligible:
        reasons.append("forward_paper_evidence_not_ready")

    eligible = not reasons
    return DeploymentEvidenceReport(
        research_ready=bool(research_ready),
        research_grade=research_grade,
        quarantine_audit_passed=audit_passed,
        paper_live_evidence_eligible=paper_eligible,
        hard_risk_engine_required=True,
        broker_execution_available=False,
        eligible_for_manual_live_review=eligible,
        reasons=tuple(reasons),
    )
