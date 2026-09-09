from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from stockbot.data.schemas import DataGrade
from stockbot.paper.arena import PaperArenaReport
from stockbot.research.deep_feedback import build_research_cycle_id
from stockbot.research.jobs import make_job_manifest
from stockbot.research.quarantine_audit import QuarantineAuditRecord, QuarantineAuditResult


@dataclass(frozen=True)
class VerifiedResearchEvidence:
    research_ready: bool
    data_grade: DataGrade
    job_id: str
    experiment_id: str
    research_cycle_id: str
    dataset_fingerprint: str
    quarantine_start: str | None
    quality_fingerprint: str


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


def _load_json_object(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _quality_fingerprint(payload: dict) -> str:
    canonical = {
        key: value
        for key, value in payload.items()
        if key not in {"quality_fingerprint", "declared_grade", "effective_grade"}
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _expected_effective_grade(payload: dict) -> DataGrade:
    try:
        declared = DataGrade(str(payload["declared_grade"]))
        persisted = DataGrade(str(payload["effective_grade"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("research data grade evidence is invalid") from exc
    eligible = bool(payload.get("research_grade_eligible", False))
    if declared is DataGrade.RESEARCH_GRADE:
        expected = DataGrade.RESEARCH_GRADE if eligible else DataGrade.BOOTSTRAP
    else:
        expected = declared
    if persisted is not expected:
        raise ValueError("research effective data grade mismatch")
    return persisted


def verify_research_evidence_bundle(
    run_dir: str | Path,
    *,
    artifact_manifest: object,
) -> VerifiedResearchEvidence:
    """Verify persisted research evidence is the exact lineage frozen for paper trading.

    This verifies identity and readiness only. It never grants execution authority.
    """

    root = Path(run_dir)
    job = _load_json_object(root / "job.json", "research job manifest")
    summary = _load_json_object(root / "summary.json", "research run summary")
    cycle = _load_json_object(root / "research_cycle.json", "research cycle manifest")
    quality = _load_json_object(root / "data_quality.json", "research data quality evidence")

    try:
        if int(job.get("schema_version", -1)) != 4:
            raise ValueError("unsupported research job schema")
        if int(summary.get("schema_version", -1)) != 3:
            raise ValueError("unsupported research summary schema")
        if int(cycle.get("schema_version", -1)) != 1:
            raise ValueError("unsupported research cycle schema")

        expected_job = make_job_manifest(
            snapshot_id=str(job["snapshot_id"]),
            dataset_fingerprint=str(job["dataset_fingerprint"]),
            horizons=tuple(int(value) for value in job["horizons"]),
            max_candidates=int(job["max_candidates"]),
            max_workers=int(job["max_workers"]),
            memory_path=job.get("memory_path"),
            quarantine_start=job.get("quarantine_start"),
            auxiliary_fingerprint=job.get("auxiliary_fingerprint"),
            auxiliary_features=tuple(job.get("auxiliary_features", ())),
            universe_fingerprint=job.get("universe_fingerprint"),
            quality_fingerprint=job.get("quality_fingerprint"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("research job identity evidence is invalid") from exc
    if expected_job.job_id != str(job.get("job_id", "")):
        raise ValueError("research job identity mismatch")
    if str(summary.get("job_id", "")) != expected_job.job_id:
        raise ValueError("research summary job identity mismatch")

    expected_quality_fingerprint = _quality_fingerprint(quality)
    persisted_quality_fingerprint = str(quality.get("quality_fingerprint", ""))
    if not persisted_quality_fingerprint or persisted_quality_fingerprint != expected_quality_fingerprint:
        raise ValueError("research quality fingerprint mismatch")
    data_grade = _expected_effective_grade(quality)

    try:
        expected_cycle_id = build_research_cycle_id(
            dataset_fingerprint=str(cycle["dataset_fingerprint"]),
            quarantine_start=cycle.get("quarantine_start"),
            universe_fingerprint=cycle.get("universe_fingerprint"),
            auxiliary_fingerprint=cycle.get("auxiliary_fingerprint"),
            quality_fingerprint=cycle.get("quality_fingerprint"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("research cycle identity evidence is invalid") from exc
    research_cycle_id = str(cycle.get("research_cycle_id", ""))
    if research_cycle_id != expected_cycle_id:
        raise ValueError("research cycle identity mismatch")

    bindings = (
        ("dataset", job.get("dataset_fingerprint"), cycle.get("dataset_fingerprint")),
        ("quarantine", job.get("quarantine_start"), cycle.get("quarantine_start")),
        ("universe", job.get("universe_fingerprint"), cycle.get("universe_fingerprint")),
        ("auxiliary", job.get("auxiliary_fingerprint"), cycle.get("auxiliary_fingerprint")),
        ("quality", job.get("quality_fingerprint"), cycle.get("quality_fingerprint")),
    )
    for label, job_value, cycle_value in bindings:
        if job_value != cycle_value:
            raise ValueError(f"research {label} evidence binding mismatch")
    if cycle.get("quality_fingerprint") != expected_quality_fingerprint:
        raise ValueError("research quality fingerprint binding mismatch")

    artifact_experiment_id = str(getattr(artifact_manifest, "experiment_id", ""))
    artifact_cycle_id = str(getattr(artifact_manifest, "research_cycle_id", ""))
    artifact_dataset_id = str(getattr(artifact_manifest, "source_dataset_fingerprint", ""))
    if not artifact_experiment_id or not artifact_cycle_id or not artifact_dataset_id:
        raise ValueError("frozen artifact research lineage is incomplete")
    if str(summary.get("champion_experiment_id", "")) != artifact_experiment_id:
        raise ValueError("research champion experiment does not match frozen artifact experiment")
    if research_cycle_id != artifact_cycle_id:
        raise ValueError("research cycle does not match frozen artifact cycle")
    dataset_fingerprint = str(cycle.get("dataset_fingerprint", ""))
    if dataset_fingerprint != artifact_dataset_id:
        raise ValueError("research dataset does not match frozen artifact dataset")

    readiness_reasons = tuple(
        str(value)
        for value in (summary.get("champion_paper_readiness_reasons", ()) or ())
    )
    research_ready = bool(summary.get("champion_paper_ready", False)) and not readiness_reasons
    return VerifiedResearchEvidence(
        research_ready=research_ready,
        data_grade=data_grade,
        job_id=expected_job.job_id,
        experiment_id=artifact_experiment_id,
        research_cycle_id=research_cycle_id,
        dataset_fingerprint=dataset_fingerprint,
        quarantine_start=(
            None if cycle.get("quarantine_start") is None else str(cycle.get("quarantine_start"))
        ),
        quality_fingerprint=expected_quality_fingerprint,
    )


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


def _quarantine_audit_passed(
    quarantine_audit: QuarantineAuditResult | None,
    quarantine_audit_record: QuarantineAuditRecord | None,
) -> bool:
    if quarantine_audit is not None and quarantine_audit_record is not None:
        raise ValueError("provide exactly one quarantine audit evidence source, not both")
    if quarantine_audit_record is not None:
        return bool(getattr(quarantine_audit_record, "passed", False))
    if quarantine_audit is None:
        return False
    holdout_report = getattr(quarantine_audit, "holdout_report", None)
    return holdout_report is not None and bool(getattr(holdout_report, "passed", False))


def evaluate_deployment_evidence(
    *,
    research_ready: bool,
    data_grade: DataGrade,
    quarantine_audit: QuarantineAuditResult | None,
    paper_report: PaperArenaReport | None,
    paper_provenance_report: object | None = None,
    quarantine_audit_record: QuarantineAuditRecord | None = None,
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
    audit_passed = _quarantine_audit_passed(quarantine_audit, quarantine_audit_record)
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
