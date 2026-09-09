from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from stockbot.ml.models import ModelConfig
from stockbot.research.holdout import HoldoutConfig, HoldoutReport, evaluate_blind_holdout
from stockbot.research.quarantine import QuarantineConfig, QuarantineManifest, split_sealed_quarantine


_AUDIT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class FrozenStrategySpec:
    strategy_id: str
    experiment_id: str
    horizon: int
    model_name: str
    model_params: dict[str, Any]
    seed: int
    top_fraction: float
    weighting: str


@dataclass(frozen=True)
class QuarantineAuditRecord:
    audit_id: str
    quarantine_start: str
    quarantine_id: str
    strategy_id: str
    experiment_id: str
    score: float
    passed: bool
    reasons: tuple[str, ...]
    audited_at: str
    research_cycle_id: str = ""
    schema_version: int = _AUDIT_SCHEMA_VERSION


@dataclass(frozen=True)
class QuarantineAuditResult:
    spec: FrozenStrategySpec
    manifest: QuarantineManifest
    holdout_report: HoldoutReport
    record: QuarantineAuditRecord


def freeze_strategy_spec(
    candidate: Any,
    *,
    top_fraction: float = 0.30,
    weighting: str = "equal",
) -> FrozenStrategySpec:
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0,1]")
    if weighting not in {"equal", "conviction"}:
        raise ValueError("weighting must be equal or conviction")
    payload = {
        "experiment_id": str(candidate.experiment_id),
        "horizon": int(candidate.horizon),
        "model_name": str(candidate.model_name),
        "model_params": dict(candidate.model_params),
        "seed": int(candidate.seed),
        "top_fraction": float(top_fraction),
        "weighting": weighting,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    strategy_id = hashlib.sha256(raw).hexdigest()[:24]
    return FrozenStrategySpec(strategy_id=strategy_id, **payload)


def _audit_identity_payload(
    *,
    quarantine_start: str,
    quarantine_id: str,
    strategy_id: str,
    experiment_id: str,
    score: float,
    passed: bool,
    reasons: tuple[str, ...],
    audited_at: str,
    research_cycle_id: str = "",
) -> dict[str, Any]:
    return {
        "quarantine_start": str(quarantine_start),
        "quarantine_id": str(quarantine_id),
        "strategy_id": str(strategy_id),
        "experiment_id": str(experiment_id),
        "score": float(score),
        "passed": bool(passed),
        "reasons": tuple(str(value) for value in reasons),
        "audited_at": str(audited_at),
        "research_cycle_id": str(research_cycle_id),
    }


def _audit_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _verify_audit_record(record: QuarantineAuditRecord) -> None:
    if int(record.schema_version) != _AUDIT_SCHEMA_VERSION:
        raise ValueError("quarantine audit integrity violation: unsupported schema")
    identity = _audit_identity_payload(
        quarantine_start=record.quarantine_start,
        quarantine_id=record.quarantine_id,
        strategy_id=record.strategy_id,
        experiment_id=record.experiment_id,
        score=record.score,
        passed=record.passed,
        reasons=record.reasons,
        audited_at=record.audited_at,
        research_cycle_id=record.research_cycle_id,
    )
    if _audit_id(identity) != str(record.audit_id):
        raise ValueError("quarantine audit integrity violation: audit ID mismatch")


def _load_ledger(path: str | Path) -> list[QuarantineAuditRecord]:
    target = Path(path)
    if not target.exists():
        return []
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("quarantine audit integrity violation: invalid ledger JSON") from exc
    if not isinstance(payload, list):
        raise ValueError("quarantine audit ledger must contain a JSON array")

    records: list[QuarantineAuditRecord] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("quarantine audit integrity violation: audit record must be an object")
        if type(item.get("passed")) is not bool:
            raise ValueError("quarantine audit integrity violation: audit passed must be boolean")
        try:
            record = QuarantineAuditRecord(
                **{
                    **item,
                    "reasons": tuple(item.get("reasons", ())),
                }
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("quarantine audit integrity violation: invalid audit record") from exc
        _verify_audit_record(record)
        records.append(record)
    return records


def load_verified_quarantine_audit_record(
    path: str | Path,
    *,
    quarantine_start: str,
    strategy_id: str,
    research_cycle_id: str | None = None,
) -> QuarantineAuditRecord:
    """Return the integrity-verified audit record for one sealed cycle and strategy."""

    records = _load_ledger(path)
    same_cycle = [
        record
        for record in records
        if pd.Timestamp(record.quarantine_start) == pd.Timestamp(quarantine_start)
    ]
    if not same_cycle:
        raise ValueError("verified quarantine audit not found for sealed cycle")
    if len(same_cycle) != 1:
        raise ValueError("multiple verified quarantine audits found for sealed cycle")
    record = same_cycle[0]
    if record.strategy_id != str(strategy_id):
        raise ValueError("verified quarantine audit strategy mismatch")
    if research_cycle_id is not None and record.research_cycle_id != str(research_cycle_id).strip():
        raise ValueError("verified quarantine audit research cycle mismatch")
    return record


def _write_ledger(path: str | Path, records: list[QuarantineAuditRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps([asdict(record) for record in records], sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _assert_audit_available(
    records: list[QuarantineAuditRecord],
    quarantine_start: str,
    strategy_id: str,
) -> QuarantineAuditRecord | None:
    same_cycle = [record for record in records if pd.Timestamp(record.quarantine_start) == pd.Timestamp(quarantine_start)]
    if not same_cycle:
        return None
    existing = same_cycle[0]
    if existing.strategy_id != strategy_id:
        raise ValueError(
            "sealed quarantine cycle has already audited another strategy; "
            "do not turn the audit set into a leaderboard"
        )
    return existing


def run_single_quarantine_audit(
    bars: pd.DataFrame,
    quarantine_config: QuarantineConfig,
    spec: FrozenStrategySpec,
    *,
    ledger_path: str | Path,
    holdout_config: HoldoutConfig | None = None,
    research_cycle_id: str | None = None,
) -> QuarantineAuditResult:
    """Audit exactly one frozen strategy per sealed quarantine research cycle.

    The function never performs candidate search. A persistent ledger refuses a second
    different strategy against the same quarantine boundary, preventing repeated audit
    probing from becoming another hyperparameter-selection loop. Persisted audit records
    are identity-verified before reuse so a changed pass/fail result fails closed.
    """

    cycle_id = "" if research_cycle_id is None else str(research_cycle_id).strip()
    if research_cycle_id is not None and not cycle_id:
        raise ValueError("research_cycle_id cannot be empty")

    split = split_sealed_quarantine(bars, quarantine_config)
    records = _load_ledger(ledger_path)
    existing = _assert_audit_available(records, split.manifest.start, spec.strategy_id)
    if existing is not None:
        raise ValueError("this frozen strategy has already been audited on the sealed quarantine cycle")

    model = ModelConfig(spec.model_name, params=dict(spec.model_params), seed=spec.seed)
    report = evaluate_blind_holdout(
        bars,
        model,
        horizon=spec.horizon,
        holdout_start=pd.Timestamp(split.manifest.start),
        config=holdout_config,
        top_fraction=spec.top_fraction,
        weighting=spec.weighting,
    )
    audited_at = datetime.now(timezone.utc).isoformat()
    identity = _audit_identity_payload(
        quarantine_start=split.manifest.start,
        quarantine_id=split.manifest.quarantine_id,
        strategy_id=spec.strategy_id,
        experiment_id=spec.experiment_id,
        score=float(report.score),
        passed=bool(report.passed),
        reasons=tuple(report.reasons),
        audited_at=audited_at,
        research_cycle_id=cycle_id,
    )
    record = QuarantineAuditRecord(
        audit_id=_audit_id(identity),
        **identity,
    )
    records.append(record)
    _write_ledger(ledger_path, records)
    return QuarantineAuditResult(
        spec=spec,
        manifest=split.manifest,
        holdout_report=report,
        record=record,
    )
