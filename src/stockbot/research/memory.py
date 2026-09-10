from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from stockbot.ml.models import ModelConfig


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    model_name: str
    model_params: dict[str, Any]
    seed: int
    dataset_fingerprint: str
    horizon: int
    factory_score: float
    base_score: float
    robustness: float
    oos_coverage: float
    metrics: dict[str, float]
    passed_gates: bool
    rejection_reasons: tuple[str, ...]
    created_at: str
    stress_score: float = 1.0


def experiment_id(
    model: ModelConfig,
    *,
    dataset_fingerprint: str,
    horizon: int,
) -> str:
    payload = {
        "name": model.name,
        "params": dict(sorted(dict(model.params).items())),
        "seed": int(model.seed),
        "dataset_fingerprint": dataset_fingerprint,
        "horizon": int(horizon),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def make_record(
    model: ModelConfig,
    *,
    dataset_fingerprint: str,
    horizon: int,
    factory_score: float,
    base_score: float,
    robustness: float,
    oos_coverage: float,
    metrics: dict[str, float],
    passed_gates: bool,
    rejection_reasons: Iterable[str] = (),
    stress_score: float = 1.0,
) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=experiment_id(model, dataset_fingerprint=dataset_fingerprint, horizon=horizon),
        model_name=model.name,
        model_params=dict(model.params),
        seed=int(model.seed),
        dataset_fingerprint=str(dataset_fingerprint),
        horizon=int(horizon),
        factory_score=float(factory_score),
        base_score=float(base_score),
        robustness=float(robustness),
        oos_coverage=float(oos_coverage),
        metrics={key: float(value) for key, value in metrics.items()},
        passed_gates=bool(passed_gates),
        rejection_reasons=tuple(str(reason) for reason in rejection_reasons),
        created_at=datetime.now(timezone.utc).isoformat(),
        stress_score=float(stress_score),
    )


class JsonlExperimentMemory:
    """Append-only, cached research memory.

    JSONL keeps the first implementation dependency-free and auditable. Records and
    experiment IDs are loaded once per process so a 500+ candidate factory run does
    not repeatedly reread the full file while checking duplicates.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records_cache: list[ExperimentRecord] | None = None
        self._seen_ids: set[str] | None = None

    def _load(self) -> list[ExperimentRecord]:
        if self._records_cache is not None:
            return self._records_cache
        rows: list[ExperimentRecord] = []
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    payload["rejection_reasons"] = tuple(payload.get("rejection_reasons", ()))
                    rows.append(ExperimentRecord(**payload))
        self._records_cache = rows
        self._seen_ids = {row.experiment_id for row in rows}
        return rows

    def append(self, record: ExperimentRecord) -> None:
        self._load()
        assert self._records_cache is not None
        assert self._seen_ids is not None
        if record.experiment_id in self._seen_ids:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(record)
        payload["rejection_reasons"] = list(record.rejection_reasons)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        self._records_cache.append(record)
        self._seen_ids.add(record.experiment_id)

    def records(self) -> list[ExperimentRecord]:
        return list(self._load())

    def seen(self, experiment_id_value: str) -> bool:
        self._load()
        assert self._seen_ids is not None
        return experiment_id_value in self._seen_ids

    def best(self, *, only_passed: bool = True, limit: int = 20) -> list[ExperimentRecord]:
        if limit <= 0:
            return []
        rows = list(self._load())
        if only_passed:
            rows = [row for row in rows if row.passed_gates]
        rows.sort(key=lambda row: row.factory_score, reverse=True)
        return rows[:limit]
