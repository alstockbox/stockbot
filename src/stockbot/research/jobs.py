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
    schema_version: int = 1


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
    completed_at: str
    schema_version: int = 1


def make_job_manifest(
    *,
    snapshot_id: str,
    dataset_fingerprint: str,
    horizons: tuple[int, ...],
    max_candidates: int,
    max_workers: int,
    memory_path: str | None,
) -> ResearchJobManifest:
    payload = {
        "snapshot_id": snapshot_id,
        "dataset_fingerprint": dataset_fingerprint,
        "horizons": tuple(int(value) for value in horizons),
        "max_candidates": int(max_candidates),
        "max_workers": int(max_workers),
        "memory_path": memory_path,
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
    )


def make_run_summary(job_id: str, report: Any) -> ResearchRunSummary:
    champion = getattr(report, "champion_candidate", None)
    active = getattr(report, "active_champion", None)
    ensemble = getattr(report, "ensemble_report", None)
    return ResearchRunSummary(
        job_id=str(job_id),
        experiments_run=int(report.experiments_run),
        candidates_passed=int(report.candidates_passed),
        holdout_evaluated=int(report.holdout_evaluated),
        promotion_occurred=bool(report.promotion_occurred),
        champion_experiment_id=(None if champion is None else str(champion.experiment_id)),
        active_champion_experiment_id=(None if active is None else str(active.experiment_id)),
        ensemble_score=(None if ensemble is None else float(ensemble.score)),
        completed_at=datetime.now(timezone.utc).isoformat(),
    )


def write_json_record(path: str | Path, record: ResearchJobManifest | ResearchRunSummary) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = asdict(record)
    if "horizons" in payload:
        payload["horizons"] = list(payload["horizons"])
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
