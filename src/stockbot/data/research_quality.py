from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockbot.data.market_schema import validate_canonical_bars
from stockbot.data.schemas import DataGrade
from stockbot.data.universe import PointInTimeUniverse, UniverseCoverageReport


@dataclass(frozen=True)
class ResearchDataAttestation:
    """Explicit provider/import-pipeline attestations that columns alone cannot prove."""

    adjusted_prices_verified: bool = False
    corporate_actions_complete: bool = False
    corporate_actions_point_in_time: bool = False


@dataclass(frozen=True)
class ResearchDataQualityCriteria:
    min_adjusted_coverage: float = 0.99
    min_retrieval_causality: float = 1.0
    min_universe_coverage: float = 0.99
    require_point_in_time_universe: bool = True
    require_survivorship_control: bool = True
    require_delisted_securities: bool = True
    require_adjusted_prices_verified: bool = True
    require_corporate_actions_complete: bool = True
    require_corporate_actions_point_in_time: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("min_adjusted_coverage", self.min_adjusted_coverage),
            ("min_retrieval_causality", self.min_retrieval_causality),
            ("min_universe_coverage", self.min_universe_coverage),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")


@dataclass(frozen=True)
class ResearchDataQualityReport:
    canonical_schema_valid: bool
    adjusted_coverage: float
    retrieval_causality_fraction: float
    provider_count: int
    providers: tuple[str, ...]
    corporate_action_fields_present: bool
    universe_report: UniverseCoverageReport | None
    attestation: ResearchDataAttestation
    research_grade_eligible: bool
    reasons: tuple[str, ...]


def evaluate_research_data_quality(
    bars: pd.DataFrame,
    *,
    universe: PointInTimeUniverse | None = None,
    attestation: ResearchDataAttestation | None = None,
    criteria: ResearchDataQualityCriteria | None = None,
) -> ResearchDataQualityReport:
    """Verify whether canonical market data is eligible for research-grade promotion.

    This function deliberately separates *observable* checks from explicit provider /
    ingestion attestations. The presence of adjusted-price or corporate-action columns
    does not prove that their historical values are complete or point-in-time correct.
    Missing evidence fails closed.
    """

    cfg = criteria or ResearchDataQualityCriteria()
    declared = attestation or ResearchDataAttestation()
    validate_canonical_bars(bars)

    adjusted_columns = ("adj_open", "adj_high", "adj_low", "adj_close", "adj_volume")
    adjusted = bars.loc[:, adjusted_columns].apply(pd.to_numeric, errors="coerce")
    finite_adjusted = adjusted.replace([np.inf, -np.inf], np.nan).notna()
    adjusted_coverage = float(finite_adjusted.to_numpy().mean()) if len(bars) else 0.0

    timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="raise")
    retrieved = pd.to_datetime(bars["retrieved_at"], utc=True, errors="raise")
    retrieval_causality = float((retrieved >= timestamps).mean()) if len(bars) else 0.0

    providers = tuple(sorted({str(value).strip() for value in bars["provider"] if str(value).strip()}))
    corporate_action_fields_present = bool(
        bars["div_cash"].notna().all()
        and bars["split_factor"].notna().all()
        and np.isfinite(pd.to_numeric(bars["div_cash"], errors="coerce")).all()
        and np.isfinite(pd.to_numeric(bars["split_factor"], errors="coerce")).all()
    )

    universe_report = None
    if universe is not None:
        universe_report = universe.coverage_for_bars(
            bars,
            min_membership_coverage=cfg.min_universe_coverage,
        )

    reasons: list[str] = []
    if adjusted_coverage < cfg.min_adjusted_coverage:
        reasons.append("insufficient_adjusted_price_coverage")
    if retrieval_causality < cfg.min_retrieval_causality:
        reasons.append("non_causal_retrieval_timestamps")
    if not corporate_action_fields_present:
        reasons.append("corporate_action_fields_incomplete")

    if cfg.require_adjusted_prices_verified and not declared.adjusted_prices_verified:
        reasons.append("adjusted_prices_not_verified")
    if cfg.require_corporate_actions_complete and not declared.corporate_actions_complete:
        reasons.append("corporate_actions_not_attested_complete")
    if cfg.require_corporate_actions_point_in_time and not declared.corporate_actions_point_in_time:
        reasons.append("corporate_actions_not_point_in_time_attested")

    if universe_report is None:
        if cfg.require_point_in_time_universe or cfg.require_survivorship_control or cfg.require_delisted_securities:
            reasons.append("point_in_time_universe_missing")
    else:
        if universe_report.membership_coverage < cfg.min_universe_coverage:
            reasons.append("insufficient_point_in_time_membership_coverage")
        if cfg.require_point_in_time_universe and not universe_report.point_in_time_membership:
            reasons.append("membership_not_point_in_time")
        if cfg.require_survivorship_control and not universe_report.survivorship_bias_controlled:
            reasons.append("survivorship_bias_not_controlled")
        if cfg.require_delisted_securities and not universe_report.includes_delisted_securities:
            reasons.append("delisted_securities_not_included")

    # Preserve deterministic ordering while avoiding duplicate reasons also emitted by
    # the nested universe report.
    deduped = tuple(dict.fromkeys(reasons))
    return ResearchDataQualityReport(
        canonical_schema_valid=True,
        adjusted_coverage=adjusted_coverage,
        retrieval_causality_fraction=retrieval_causality,
        provider_count=len(providers),
        providers=providers,
        corporate_action_fields_present=corporate_action_fields_present,
        universe_report=universe_report,
        attestation=declared,
        research_grade_eligible=not deduped,
        reasons=deduped,
    )


def verified_data_grade(
    declared_grade: DataGrade,
    quality_report: ResearchDataQualityReport | None,
) -> DataGrade:
    """Fail closed when a snapshot claims RESEARCH_GRADE without verifiable evidence."""

    if declared_grade is not DataGrade.RESEARCH_GRADE:
        return declared_grade
    if quality_report is None or not quality_report.research_grade_eligible:
        return DataGrade.BOOTSTRAP
    return DataGrade.RESEARCH_GRADE
