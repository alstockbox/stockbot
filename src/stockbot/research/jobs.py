from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResearchJobManifest:
    job_id: str
    snapshot_id: str
    dataset_fingerprint: str
    horizons: tuple[int, ...]
    max_candidates: int
    max_workers: int
    memory_path: str | None
    created_at: str
    quarantine_start: str | None = None
    auxiliary_fingerprint: str | None = None
    auxiliary_features: tuple[str, ...] = ()
    universe_fingerprint: str | None = None
    schema_version: int = 3


@dataclass(frozen=True)
class ResearchRunSummary:
    job_id: str
    experiments_run: int
    candidates_passed: int
    holdout_evaluated: int
    promotion_occurred: bool
    champion_experiment_id: str | None
    active_champion_experiment_id: str | None
    ensemble_score: float | None
    specialist_router_score: float | None
    specialist_scores: dict[str, float]
    specialist_candidate_pairs_tested: int
    completed_at: str
    schema_version: int = 2


def make_job_manifest(
    *,
    snapshot_id: str,
    dataset_fingerprint: str,
    horizons: tuple[int, ...],
    max_candidates: int,
    max_workers: int,
    memory_path: str | None,
    quarantine_start: str | None = None,
    auxiliary_fingerprint: str | None = None,
    auxiliary_features: tuple[str, ...] | list[str] = (),
    universe_fingerprint: str | None = None,
) -> ResearchJobManifest:
    normalized_auxiliary = tuple(sorted({str(value).strip() for value in auxiliary_features if str(value).strip()}))
    payload = {
        "snapshot_id": snapshot_id,
        "dataset_fingerprint": dataset_fingerprint,
        "horizons": tuple(int(value) for value in horizons),
        "max_candidates": int(max_candidates),
        "max_workers": int(max_workers),
        "memory_path": memory_path,
        "quarantine_start": quarantine_start,
        "auxiliary_fingerprint": auxiliary_fingerprint,
        "auxiliary_features": normalized_auxiliary,
        "universe_fingerprint": universe_fingerprint,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    job_id = hashlib.sha256(raw).hexdigest()[:24]
    return ResearchJobManifest(
        job_id=job_id,
        snapshot_id=str(snapshot_id),
        dataset_fingerprint=str(dataset_fingerprint),
        horizons=tuple(int(value) for value in horizons),
        max_candidates=int(max_candidates),
        max_workers=int(max_workers),
        memory_path=memory_path,
        created_at=datetime.now(timezone.utc).isoformat(),
        quarantine_start=quarantine_start,
        auxiliary_fingerprint=auxiliary_fingerprint,
        auxiliary_features=normalized_auxiliary,
        universe_fingerprint=universe_fingerprint,
    )


def make_run_summary(
    job_id: str,
    report: Any,
    specialist_diagnostics: Any | None = None,
) -> ResearchRunSummary:
    champion = getattr(report, "champion_candidate", None)
    active = getattr(report, "active_champion", None)
    ensemble = getattr(report, "ensemble_report", None)

    router = None
    specialists: dict[str, Any] = {}
    pairs_tested = 0
    if specialist_diagnostics is not None:
        router = getattr(specialist_diagnostics, "router_report", None)
        specialists = dict(getattr(specialist_diagnostics, "specialists", {}) or {})
        pairs_tested = int(getattr(specialist_diagnostics, "candidate_pairs_tested", 0))

    return ResearchRunSummary(
        job_id=str(job_id),
        experiments_run=int(report.experiments_run),
        candidates_passed=int(report.candidates_passed),
        holdout_evaluated=int(report.holdout_evaluated),
        promotion_occurred=bool(report.promotion_occurred),
        champion_experiment_id=(None if champion is None else str(champion.experiment_id)),
        active_champion_experiment_id=(None if active is None else str(active.experiment_id)),
        ensemble_score=(None if ensemble is None else float(ensemble.score)),
        specialist_router_score=(None if router is None else float(router.score)),
        specialist_scores={key: float(value.score) for key, value in specialists.items()},
        specialist_candidate_pairs_tested=pairs_tested,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )


def _atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def write_json_record(path: str | Path, record: ResearchJobManifest | ResearchRunSummary) -> None:
    payload = asdict(record)
    if "horizons" in payload:
        payload["horizons"] = list(payload["horizons"])
    if "auxiliary_features" in payload:
        payload["auxiliary_features"] = list(payload["auxiliary_features"])
    _atomic_write_json(path, payload)


def write_json_payload(path: str | Path, payload: Any) -> None:
    """Atomically persist an already JSON-friendly research artifact."""

    _atomic_write_json(path, payload)
