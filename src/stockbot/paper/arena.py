from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockbot.evaluation.metrics import performance_metrics
from stockbot.paper.ledger import PaperObservation
from stockbot.research.bootstrap_uncertainty import BootstrapUncertaintyReport, evaluate_block_bootstrap_uncertainty
from stockbot.research.drift import DriftReport, evaluate_return_drift
from stockbot.research.stress import StressReport, evaluate_stress_suite


@dataclass(frozen=True)
class PaperArenaCriteria:
    min_sessions: int = 60
    min_live_span_days: int = 45
    min_sharpe: float = 0.25
    min_excess_cagr: float = 0.0
    max_drawdown: float = 0.20
    min_fill_rate: float = 0.90
    max_average_turnover: float = 2.0
    min_bootstrap_confidence: float = 0.70
    reject_drift: bool = True
    require_frozen_provenance: bool = False

    def __post_init__(self) -> None:
        if self.min_sessions <= 0 or self.min_live_span_days <= 0:
            raise ValueError("paper minimum sessions/span must be positive")
        if not 0.0 <= self.min_fill_rate <= 1.0:
            raise ValueError("min_fill_rate must be in [0,1]")
        if not 0.0 <= self.min_bootstrap_confidence <= 1.0:
            raise ValueError("min_bootstrap_confidence must be in [0,1]")
        if self.max_drawdown < 0.0 or self.max_average_turnover < 0.0:
            raise ValueError("paper risk limits must be non-negative")


@dataclass(frozen=True)
class PaperArenaReport:
    strategy_id: str
    sessions: int
    live_span_days: int
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    excess_cagr: float
    average_turnover: float
    average_cost_rate: float
    average_fill_rate: float
    stress_report: StressReport
    drift_report: DriftReport
    bootstrap_report: BootstrapUncertaintyReport | None
    evidence_score: float
    live_eligible: bool
    reasons: tuple[str, ...]
    model_artifact_id: str | None = None
    research_cycle_id: str | None = None
    frozen_provenance_complete: bool = False


def _single_optional_lineage(
    observations: list[PaperObservation] | tuple[PaperObservation, ...],
    *,
    attribute: str,
    label: str,
) -> str | None:
    raw = [getattr(row, attribute) for row in observations]
    present = [str(value).strip() for value in raw if value is not None and str(value).strip()]
    if not present:
        return None
    if len(present) != len(raw):
        raise ValueError(f"paper track contains partial {label} lineage")
    unique = set(present)
    if len(unique) != 1:
        raise ValueError(f"paper track contains mixed {label} lineage")
    return present[0]


def _frozen_provenance(
    observations: list[PaperObservation] | tuple[PaperObservation, ...],
) -> tuple[str | None, str | None, bool]:
    artifact_id = _single_optional_lineage(
        observations,
        attribute="model_artifact_id",
        label="model artifact",
    )
    cycle_id = _single_optional_lineage(
        observations,
        attribute="research_cycle_id",
        label="research cycle",
    )
    complete = bool(
        artifact_id
        and cycle_id
        and all(row.signal_timestamp for row in observations)
        and all(row.signal_snapshot_fingerprint for row in observations)
        and all(row.realization_snapshot_fingerprint for row in observations)
    )
    return artifact_id, cycle_id, complete


def evaluate_paper_track(
    observations: list[PaperObservation] | tuple[PaperObservation, ...],
    *,
    criteria: PaperArenaCriteria | None = None,
) -> PaperArenaReport:
    """Evaluate forward paper evidence without granting any broker authority.

    Generic research/synthetic tracks may omit frozen-runner provenance. Explicit
    artifact/cycle lineage, however, may never be partial or mixed under one strategy.
    Operational/deployment callers can require full frozen provenance: one artifact,
    one research cycle, signal time, signal snapshot and realization snapshot on every
    observation.
    """

    if not observations:
        raise ValueError("paper track requires observations")
    ids = {row.strategy_id for row in observations}
    if len(ids) != 1:
        raise ValueError("paper track must contain exactly one strategy_id")
    artifact_id, cycle_id, frozen_provenance_complete = _frozen_provenance(observations)
    cfg = criteria or PaperArenaCriteria()
    ordered = sorted(observations, key=lambda row: row.timestamp)
    timestamps = pd.to_datetime([row.timestamp for row in ordered], utc=True)
    if timestamps.duplicated().any():
        raise ValueError("paper track contains duplicate timestamps")

    returns = pd.Series([row.net_return for row in ordered], index=timestamps, dtype=float)
    benchmark = pd.Series([row.benchmark_return for row in ordered], index=timestamps, dtype=float)
    turnover = pd.Series([row.turnover for row in ordered], index=timestamps, dtype=float)
    costs = pd.Series([row.cost_rate for row in ordered], index=timestamps, dtype=float)
    fills = pd.Series([row.fill_rate for row in ordered], index=timestamps, dtype=float)

    metrics = performance_metrics(returns)
    benchmark_metrics = performance_metrics(benchmark)
    excess_cagr = float(metrics.get("cagr", 0.0) - benchmark_metrics.get("cagr", 0.0))
    stress = evaluate_stress_suite(returns, turnover=turnover)
    drift = evaluate_return_drift(returns)
    bootstrap = None
    if len(returns) >= 20:
        bootstrap = evaluate_block_bootstrap_uncertainty(
            returns,
            block_size=min(10, len(returns)),
            samples=200,
            confidence_level=0.90,
            seed=20260908,
        )

    sessions = len(ordered)
    live_span_days = int((timestamps.max() - timestamps.min()).days) if sessions > 1 else 0
    average_turnover = float(turnover.mean())
    average_cost_rate = float(costs.mean())
    average_fill_rate = float(fills.mean())
    sharpe = float(metrics.get("sharpe", 0.0))
    max_drawdown = float(metrics.get("max_drawdown", 1.0))
    bootstrap_confidence = 0.0 if bootstrap is None else bootstrap.confidence_score

    reasons: list[str] = []
    if sessions < cfg.min_sessions:
        reasons.append("insufficient_paper_sessions")
    if live_span_days < cfg.min_live_span_days:
        reasons.append("insufficient_paper_calendar_span")
    if sharpe < cfg.min_sharpe:
        reasons.append("paper_sharpe")
    if excess_cagr <= cfg.min_excess_cagr:
        reasons.append("paper_excess_cagr")
    if max_drawdown > cfg.max_drawdown:
        reasons.append("paper_drawdown")
    if average_fill_rate < cfg.min_fill_rate:
        reasons.append("paper_fill_quality")
    if average_turnover > cfg.max_average_turnover:
        reasons.append("paper_turnover")
    if bootstrap_confidence < cfg.min_bootstrap_confidence:
        reasons.append("paper_bootstrap_confidence")
    if cfg.reject_drift and drift.degraded:
        reasons.append("paper_drift")
    if cfg.require_frozen_provenance and not frozen_provenance_complete:
        reasons.append("paper_frozen_provenance")

    evidence_score = float(
        np.clip(
            0.20 * min(1.0, sessions / cfg.min_sessions)
            + 0.10 * min(1.0, live_span_days / cfg.min_live_span_days)
            + 0.15 * float(np.clip((sharpe + 0.5) / 2.0, 0.0, 1.0))
            + 0.15 * float(np.clip(0.5 + excess_cagr, 0.0, 1.0))
            + 0.10 * float(np.clip(1.0 - max_drawdown, 0.0, 1.0))
            + 0.10 * average_fill_rate
            + 0.10 * stress.score
            + 0.10 * bootstrap_confidence,
            0.0,
            1.0,
        )
    )

    return PaperArenaReport(
        strategy_id=ordered[0].strategy_id,
        sessions=sessions,
        live_span_days=live_span_days,
        metrics={key: float(value) for key, value in metrics.items()},
        benchmark_metrics={key: float(value) for key, value in benchmark_metrics.items()},
        excess_cagr=excess_cagr,
        average_turnover=average_turnover,
        average_cost_rate=average_cost_rate,
        average_fill_rate=average_fill_rate,
        stress_report=stress,
        drift_report=drift,
        bootstrap_report=bootstrap,
        evidence_score=evidence_score,
        live_eligible=not reasons,
        reasons=tuple(reasons),
        model_artifact_id=artifact_id,
        research_cycle_id=cycle_id,
        frozen_provenance_complete=frozen_provenance_complete,
    )
