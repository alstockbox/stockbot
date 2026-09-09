from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from stockbot.research.memory import ExperimentRecord


@dataclass(frozen=True)
class DeepResearchFinding:
    finding_id: str
    source_cycle_id: str
    experiment_id: str
    horizon: int
    model_name: str
    created_at: str
    adaptive_priority: float
    bootstrap_confidence: float
    factor_idiosyncratic_score: float
    factor_explained_fraction: float
    capacity_score: float
    max_effective_capital: float | None
    window_robustness_score: float
    useful_feature_groups: tuple[str, ...]
    harmful_feature_groups: tuple[str, ...]
    best_top_fraction: float
    best_weighting: str
    neutralization_score_delta: float | None
    sector_cap_score_delta: float | None
    sector_concentration_reduction: float | None
    stacking_score: float | None
    stacking_delta_vs_policy: float | None


def _bounded(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return max(0.0, min(1.0, number))


def _finite_optional(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return float(number)


def _stable_delta(left: Any, right: Any) -> float | None:
    lhs = _finite_optional(left)
    rhs = _finite_optional(right)
    if lhs is None or rhs is None:
        return None
    return round(lhs - rhs, 12)


def build_research_cycle_id(
    *,
    dataset_fingerprint: str,
    quarantine_start: str | None = None,
    universe_fingerprint: str | None = None,
    auxiliary_fingerprint: str | None = None,
    quality_fingerprint: str | None = None,
) -> str:
    """Fingerprint the data identity of a research cycle, not execution knobs.

    Worker counts, candidate budgets and scheduler paths are intentionally absent. A
    rerun over the same underlying data/control artifacts is therefore the same cycle
    and cannot consume findings produced by that data as if they were prior evidence.
    """

    dataset = str(dataset_fingerprint).strip()
    if not dataset:
        raise ValueError("dataset_fingerprint is required")
    payload = {
        "dataset_fingerprint": dataset,
        "quarantine_start": None if quarantine_start is None else str(quarantine_start),
        "universe_fingerprint": None if universe_fingerprint is None else str(universe_fingerprint),
        "auxiliary_fingerprint": None if auxiliary_fingerprint is None else str(auxiliary_fingerprint),
        "quality_fingerprint": None if quality_fingerprint is None else str(quality_fingerprint),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _finding_id(source_cycle_id: str, experiment_id: str) -> str:
    raw = f"{source_cycle_id}:{experiment_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _adaptive_priority(candidate: Any) -> float:
    """Bounded, diagnostic-only signal for future exploration allocation.

    All components are already normalized to [0,1]. Unbounded score deltas are kept as
    descriptive findings and deliberately excluded from this priority so a single
    unusually large backtest score cannot dominate future mutation allocation.
    """

    bootstrap = _bounded(getattr(candidate.bootstrap_uncertainty, "confidence_score", 0.0))
    idiosyncratic = _bounded(getattr(candidate.factor_exposure, "idiosyncratic_score", 0.0))
    capacity = _bounded(getattr(candidate.capacity_curve, "capacity_score", 0.0))
    windows = _bounded(getattr(candidate.window_robustness, "score", 0.0))
    return float(
        0.30 * bootstrap
        + 0.30 * idiosyncratic
        + 0.20 * capacity
        + 0.20 * windows
    )


def make_deep_findings(
    diagnostics: Any,
    *,
    source_cycle_id: str,
    useful_feature_threshold: float = 0.05,
) -> tuple[DeepResearchFinding, ...]:
    """Compact pre-holdout deep diagnostics into auditable future-cycle findings."""

    cycle_id = str(source_cycle_id).strip()
    if not cycle_id:
        raise ValueError("source_cycle_id is required")
    if useful_feature_threshold < 0.0:
        raise ValueError("useful_feature_threshold must be non-negative")

    rows: list[DeepResearchFinding] = []
    stacking_reports = getattr(diagnostics, "stacking_reports", {}) or {}
    for experiment_id, candidate in getattr(diagnostics, "candidates", {}).items():
        policy_result = candidate.policy_arena.best
        feature_ablation = candidate.feature_ablation
        useful = tuple(
            str(result.group)
            for result in feature_ablation.results
            if float(result.score_impact) > float(useful_feature_threshold)
        )
        harmful = tuple(str(value) for value in feature_ablation.recommended_drop_groups)

        neutralization = getattr(candidate, "neutralization", None)
        neutralization_delta = (
            None
            if neutralization is None
            else _finite_optional(getattr(neutralization, "score_delta", None))
        )

        sector_cap = getattr(candidate, "sector_cap", None)
        sector_cap_delta = (
            None
            if sector_cap is None
            else _finite_optional(getattr(sector_cap, "score_delta", None))
        )
        concentration_reduction = None
        if sector_cap is not None:
            concentration_reduction = _stable_delta(
                getattr(sector_cap.baseline_sector_exposure, "average_max_sector_weight", None),
                getattr(sector_cap.constrained_sector_exposure, "average_max_sector_weight", None),
            )

        stacking = stacking_reports.get(int(candidate.horizon))
        stacking_score = None if stacking is None else _finite_optional(getattr(stacking, "score", None))
        policy_score = _finite_optional(getattr(policy_result, "score", None))
        stacking_delta = (
            None
            if stacking_score is None or policy_score is None
            else round(stacking_score - policy_score, 12)
        )

        max_capital = _finite_optional(
            getattr(candidate.capacity_curve, "max_effective_capital", None)
        )
        rows.append(
            DeepResearchFinding(
                finding_id=_finding_id(cycle_id, str(experiment_id)),
                source_cycle_id=cycle_id,
                experiment_id=str(experiment_id),
                horizon=int(candidate.horizon),
                model_name=str(candidate.model_name),
                created_at=datetime.now(timezone.utc).isoformat(),
                adaptive_priority=_adaptive_priority(candidate),
                bootstrap_confidence=_bounded(candidate.bootstrap_uncertainty.confidence_score),
                factor_idiosyncratic_score=_bounded(candidate.factor_exposure.idiosyncratic_score),
                factor_explained_fraction=_bounded(candidate.factor_exposure.explained_fraction),
                capacity_score=_bounded(candidate.capacity_curve.capacity_score),
                max_effective_capital=max_capital,
                window_robustness_score=_bounded(candidate.window_robustness.score),
                useful_feature_groups=useful,
                harmful_feature_groups=harmful,
                best_top_fraction=float(policy_result.policy.top_fraction),
                best_weighting=str(policy_result.policy.weighting),
                neutralization_score_delta=neutralization_delta,
                sector_cap_score_delta=sector_cap_delta,
                sector_concentration_reduction=concentration_reduction,
                stacking_score=stacking_score,
                stacking_delta_vs_policy=stacking_delta,
            )
        )
    return tuple(rows)


class JsonlDeepResearchMemory:
    """Append-only deep-diagnostic memory, isolated by research-cycle identity."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records_cache: list[DeepResearchFinding] | None = None
        self._seen_ids: set[str] | None = None

    def _load(self) -> list[DeepResearchFinding]:
        if self._records_cache is not None:
            return self._records_cache
        rows: list[DeepResearchFinding] = []
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    payload["useful_feature_groups"] = tuple(payload.get("useful_feature_groups", ()))
                    payload["harmful_feature_groups"] = tuple(payload.get("harmful_feature_groups", ()))
                    rows.append(DeepResearchFinding(**payload))
        self._records_cache = rows
        self._seen_ids = {row.finding_id for row in rows}
        return rows

    def append(self, finding: DeepResearchFinding) -> None:
        self._load()
        assert self._records_cache is not None
        assert self._seen_ids is not None
        if finding.finding_id in self._seen_ids:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(finding)
        payload["useful_feature_groups"] = list(finding.useful_feature_groups)
        payload["harmful_feature_groups"] = list(finding.harmful_feature_groups)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        self._records_cache.append(finding)
        self._seen_ids.add(finding.finding_id)

    def append_many(self, findings: Iterable[DeepResearchFinding]) -> None:
        for finding in findings:
            self.append(finding)

    def records(self) -> list[DeepResearchFinding]:
        return list(self._load())

    def for_future_cycle(self, current_cycle_id: str) -> list[DeepResearchFinding]:
        cycle_id = str(current_cycle_id).strip()
        if not cycle_id:
            raise ValueError("current_cycle_id is required")
        return [row for row in self._load() if row.source_cycle_id != cycle_id]


def rank_adaptive_parent_records(
    records: Iterable[ExperimentRecord],
    findings: Iterable[DeepResearchFinding],
    *,
    current_cycle_id: str,
    limit: int,
    feedback_weight: float = 0.25,
) -> list[ExperimentRecord]:
    """Rank mutation parents using only findings from earlier data cycles.

    Deep feedback never changes a candidate's gate, factory score, holdout score or
    promotion eligibility. It only nudges future exploration allocation around prior
    experiments that also survived richer development-only diagnostics.
    """

    if limit <= 0:
        return []
    if not 0.0 <= float(feedback_weight) <= 1.0:
        raise ValueError("feedback_weight must be in [0,1]")
    cycle_id = str(current_cycle_id).strip()
    if not cycle_id:
        raise ValueError("current_cycle_id is required")

    usable_findings = [row for row in findings if row.source_cycle_id != cycle_id]
    best_finding_by_experiment: dict[str, DeepResearchFinding] = {}
    for finding in usable_findings:
        previous = best_finding_by_experiment.get(finding.experiment_id)
        if previous is None or finding.adaptive_priority > previous.adaptive_priority:
            best_finding_by_experiment[finding.experiment_id] = finding

    def score(record: ExperimentRecord) -> tuple[float, float, str]:
        finding = best_finding_by_experiment.get(record.experiment_id)
        priority = 0.5 if finding is None else _bounded(finding.adaptive_priority, 0.5)
        feedback_bonus = float(feedback_weight) * (priority - 0.5)
        combined = float(record.factory_score) + feedback_bonus
        return combined, float(record.factory_score), str(record.experiment_id)

    ranked = sorted(list(records), key=score, reverse=True)
    return ranked[:limit]
