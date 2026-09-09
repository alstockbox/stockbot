from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from stockbot.data.research_quality import ResearchDataQualityCriteria
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


def _readiness_fingerprint(summary: dict) -> str:
    try:
        raw_score = summary["champion_paper_readiness_score"]
        if raw_score is None:
            raise ValueError("research readiness score is missing")
        score = float(raw_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("research readiness score must be finite in [0,1]")
        payload = {
            "ready": bool(summary["champion_paper_ready"]),
            "score": score,
            "reasons": [
                str(value)
                for value in (summary.get("champion_paper_readiness_reasons", ()) or ())
            ],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("research readiness evidence is invalid") from exc
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _quality_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"research data quality {label} must be boolean")
    return value


def _quality_fraction(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"research data quality {label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"research data quality {label} must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"research data quality {label} must be finite in [0,1]")
    return result


def _recomputed_quality_eligibility(payload: dict) -> bool:
    """Recompute default research-grade eligibility from persisted primitive evidence."""

    cfg = ResearchDataQualityCriteria()
    if _quality_bool(payload.get("canonical_schema_valid"), "canonical_schema_valid") is not True:
        raise ValueError("research data quality canonical schema is not verified")

    adjusted_coverage = _quality_fraction(payload.get("adjusted_coverage"), "adjusted_coverage")
    retrieval_causality = _quality_fraction(
        payload.get("retrieval_causality_fraction"),
        "retrieval_causality_fraction",
    )
    corporate_action_fields_present = _quality_bool(
        payload.get("corporate_action_fields_present"),
        "corporate_action_fields_present",
    )

    attestation = payload.get("attestation")
    if not isinstance(attestation, dict):
        raise ValueError("research data quality attestation is invalid")
    adjusted_prices_verified = _quality_bool(
        attestation.get("adjusted_prices_verified"),
        "adjusted_prices_verified",
    )
    corporate_actions_complete = _quality_bool(
        attestation.get("corporate_actions_complete"),
        "corporate_actions_complete",
    )
    corporate_actions_point_in_time = _quality_bool(
        attestation.get("corporate_actions_point_in_time"),
        "corporate_actions_point_in_time",
    )

    reasons: list[str] = []
    if adjusted_coverage < cfg.min_adjusted_coverage:
        reasons.append("insufficient_adjusted_price_coverage")
    if retrieval_causality < cfg.min_retrieval_causality:
        reasons.append("non_causal_retrieval_timestamps")
    if not corporate_action_fields_present:
        reasons.append("corporate_action_fields_incomplete")
    if cfg.require_adjusted_prices_verified and not adjusted_prices_verified:
        reasons.append("adjusted_prices_not_verified")
    if cfg.require_corporate_actions_complete and not corporate_actions_complete:
        reasons.append("corporate_actions_not_attested_complete")
    if cfg.require_corporate_actions_point_in_time and not corporate_actions_point_in_time:
        reasons.append("corporate_actions_not_point_in_time_attested")

    universe = payload.get("universe_report")
    if universe is None:
        if (
            cfg.require_point_in_time_universe
            or cfg.require_survivorship_control
            or cfg.require_delisted_securities
        ):
            reasons.append("point_in_time_universe_missing")
    elif isinstance(universe, dict):
        membership_coverage = _quality_fraction(
            universe.get("membership_coverage"),
            "universe membership_coverage",
        )
        point_in_time_membership = _quality_bool(
            universe.get("point_in_time_membership"),
            "universe point_in_time_membership",
        )
        survivorship_bias_controlled = _quality_bool(
            universe.get("survivorship_bias_controlled"),
            "universe survivorship_bias_controlled",
        )
        includes_delisted_securities = _quality_bool(
            universe.get("includes_delisted_securities"),
            "universe includes_delisted_securities",
        )
        if membership_coverage < cfg.min_universe_coverage:
            reasons.append("insufficient_point_in_time_membership_coverage")
        if cfg.require_point_in_time_universe and not point_in_time_membership:
            reasons.append("membership_not_point_in_time")
        if cfg.require_survivorship_control and not survivorship_bias_controlled:
            reasons.append("survivorship_bias_not_controlled")
        if cfg.require_delisted_securities and not includes_delisted_securities:
            reasons.append("delisted_securities_not_included")
    else:
        raise ValueError("research data quality universe evidence is invalid")

    recomputed_reasons = tuple(dict.fromkeys(reasons))
    persisted_reasons = payload.get("reasons")
    if not isinstance(persisted_reasons, list):
        raise ValueError("research data quality reasons are invalid")
    if tuple(str(value) for value in persisted_reasons) != recomputed_reasons:
        raise ValueError("research data quality reasons mismatch")

    persisted_eligible = _quality_bool(
        payload.get("research_grade_eligible"),
        "research_grade_eligible",
    )
    recomputed_eligible = not recomputed_reasons
    if persisted_eligible is not recomputed_eligible:
        raise ValueError("research data quality eligibility mismatch")
    return recomputed_eligible


def _expected_effective_grade(payload: dict) -> DataGrade:
    try:
        declared = DataGrade(str(payload["declared_grade"]))
        persisted = DataGrade(str(payload["effective_grade"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("research data grade evidence is invalid") from exc
    eligible = _recomputed_quality_eligibility(payload)
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
    artifact_readiness_fingerprint = str(
        getattr(artifact_manifest, "research_readiness_fingerprint", "") or ""
    ).strip()
    if not artifact_experiment_id or not artifact_cycle_id or not artifact_dataset_id:
        raise ValueError("frozen artifact research lineage is incomplete")
    if not artifact_readiness_fingerprint:
        raise ValueError("frozen artifact research readiness fingerprint is missing")
    if _readiness_fingerprint(summary) != artifact_readiness_fingerprint:
        raise ValueError("research readiness fingerprint mismatch")
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


def _required_lineage_value(source: object, attribute: str, label: str) -> str:
    value = str(getattr(source, attribute, "") or "").strip()
    if not value:
        raise ValueError(f"deployment review requires {label}")
    return value


def evaluate_bound_deployment_review(
    *,
    research_evidence: VerifiedResearchEvidence,
    artifact_manifest: object,
    quarantine_audit_record: QuarantineAuditRecord,
    paper_report: PaperArenaReport,
    paper_provenance_report: object,
) -> DeploymentEvidenceReport:
    """Bind every independent evidence layer to one frozen artifact before review.

    The result can only express eligibility for a later manual live review. It cannot
    authorize broker execution.
    """

    if bool(getattr(artifact_manifest, "broker_execution_available", False)):
        raise ValueError("frozen artifact may not enable broker execution")
    if bool(getattr(paper_provenance_report, "broker_execution_available", False)):
        raise ValueError("paper provenance may not enable broker execution")

    artifact_strategy = _required_lineage_value(artifact_manifest, "strategy_id", "artifact strategy")
    artifact_id = _required_lineage_value(artifact_manifest, "artifact_id", "artifact ID")
    artifact_experiment = _required_lineage_value(artifact_manifest, "experiment_id", "artifact experiment")
    artifact_cycle = _required_lineage_value(artifact_manifest, "research_cycle_id", "artifact research cycle")
    artifact_dataset = _required_lineage_value(
        artifact_manifest,
        "source_dataset_fingerprint",
        "artifact source dataset",
    )

    if research_evidence.experiment_id != artifact_experiment:
        raise ValueError("research experiment does not match frozen artifact experiment")
    if research_evidence.research_cycle_id != artifact_cycle:
        raise ValueError("research cycle does not match frozen artifact cycle")
    if research_evidence.dataset_fingerprint != artifact_dataset:
        raise ValueError("research dataset does not match frozen artifact dataset")
    if research_evidence.quarantine_start is None:
        raise ValueError("deployment review requires sealed research quarantine boundary")

    audit_strategy = _required_lineage_value(quarantine_audit_record, "strategy_id", "audit strategy")
    audit_experiment = _required_lineage_value(quarantine_audit_record, "experiment_id", "audit experiment")
    audit_cycle = _required_lineage_value(
        quarantine_audit_record,
        "research_cycle_id",
        "audit research cycle",
    )
    audit_start = _required_lineage_value(quarantine_audit_record, "quarantine_start", "audit quarantine boundary")
    if audit_strategy != artifact_strategy:
        raise ValueError("audit strategy does not match frozen artifact strategy")
    if audit_experiment != artifact_experiment:
        raise ValueError("audit experiment does not match frozen artifact experiment")
    if audit_cycle != research_evidence.research_cycle_id:
        raise ValueError("audit research cycle does not match verified research cycle")
    if audit_start != research_evidence.quarantine_start:
        raise ValueError("audit quarantine boundary does not match research quarantine boundary")

    paper_strategy = _required_lineage_value(paper_report, "strategy_id", "paper strategy")
    paper_artifact = _required_lineage_value(paper_report, "model_artifact_id", "paper artifact")
    paper_cycle = _required_lineage_value(paper_report, "research_cycle_id", "paper research cycle")
    if paper_strategy != artifact_strategy:
        raise ValueError("paper strategy does not match frozen artifact strategy")
    if paper_artifact != artifact_id:
        raise ValueError("paper artifact does not match frozen artifact")
    if paper_cycle != artifact_cycle:
        raise ValueError("paper research cycle does not match frozen artifact cycle")

    provenance_strategy = _required_lineage_value(
        paper_provenance_report,
        "strategy_id",
        "provenance strategy",
    )
    provenance_artifact = _required_lineage_value(
        paper_provenance_report,
        "model_artifact_id",
        "provenance artifact",
    )
    provenance_cycle = _required_lineage_value(
        paper_provenance_report,
        "research_cycle_id",
        "provenance research cycle",
    )
    if provenance_strategy != artifact_strategy:
        raise ValueError("paper provenance strategy does not match frozen artifact strategy")
    if provenance_artifact != artifact_id:
        raise ValueError("paper provenance artifact does not match frozen artifact")
    if provenance_cycle != artifact_cycle:
        raise ValueError("paper provenance research cycle does not match frozen artifact cycle")

    return evaluate_deployment_evidence(
        research_ready=research_evidence.research_ready,
        data_grade=research_evidence.data_grade,
        quarantine_audit=None,
        quarantine_audit_record=quarantine_audit_record,
        paper_report=paper_report,
        paper_provenance_report=paper_provenance_report,
    )
