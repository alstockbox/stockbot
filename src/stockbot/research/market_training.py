from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from stockbot.data.market_schema import validate_canonical_bars
from stockbot.data.point_in_time_features import PointInTimeFeatureStore
from stockbot.data.research_quality import ResearchDataQualityReport, verified_data_grade
from stockbot.data.schemas import DatasetMetadata
from stockbot.data.snapshots import MarketSnapshot
from stockbot.data.universe import PointInTimeUniverse
from stockbot.research.champion import JsonChampionStore
from stockbot.research.deep_diagnostics import DeepResearchDiagnostics, run_deep_research_diagnostics
from stockbot.research.factory import FactoryReport, ResearchFactory, ResearchFactoryConfig
from stockbot.research.liquidity_execution import LiquidityExecutionConfig
from stockbot.research.memory import JsonlExperimentMemory
from stockbot.research.quarantine import QuarantineConfig, persist_sealed_quarantine, split_sealed_quarantine
from stockbot.research.specialist_pipeline import RegimeSpecialistDiagnostics, run_regime_specialist_diagnostics
from stockbot.research.training_pipeline import TrainingRun, run_training_research


def prepare_training_bars(canonical_bars: pd.DataFrame) -> pd.DataFrame:
    validate_canonical_bars(canonical_bars)
    result = canonical_bars.copy()
    close = pd.to_numeric(result["close"], errors="coerce")
    adj_close = pd.to_numeric(result["adj_close"], errors="coerce")
    factor = (adj_close / close).where(adj_close.notna() & close.gt(0), 1.0)

    for raw_name, adjusted_name in (
        ("open", "adj_open"),
        ("high", "adj_high"),
        ("low", "adj_low"),
        ("close", "adj_close"),
    ):
        raw = pd.to_numeric(result[raw_name], errors="coerce")
        adjusted = pd.to_numeric(result[adjusted_name], errors="coerce")
        result[raw_name] = adjusted.where(adjusted.notna(), raw * factor)

    raw_volume = pd.to_numeric(result["volume"], errors="coerce")
    adjusted_volume = pd.to_numeric(result["adj_volume"], errors="coerce")
    result["volume"] = adjusted_volume.where(adjusted_volume.notna(), raw_volume)
    keep = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    result = result.loc[:, keep].sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
    if result[["open", "high", "low", "close", "volume"]].replace([np.inf, -np.inf], np.nan).isna().any().any():
        raise ValueError("adjusted training bars contain non-finite values")
    return result


def _snapshot_metadata(
    snapshot: MarketSnapshot,
    quality_report: ResearchDataQualityReport | None = None,
) -> DatasetMetadata:
    """Create training metadata with fail-closed research-grade verification."""

    return DatasetMetadata(
        name=snapshot.snapshot_id,
        source=snapshot.manifest.provider,
        grade=verified_data_grade(snapshot.manifest.grade, quality_report),
        version=snapshot.manifest.schema_version,
        created_at=snapshot.manifest.created_at,
    )


def _development_bars(
    snapshot: MarketSnapshot,
    quarantine_config: QuarantineConfig | None,
    quarantine_manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    bars = prepare_training_bars(snapshot.bars)
    if quarantine_config is None:
        return bars
    split = split_sealed_quarantine(bars, quarantine_config)
    if quarantine_manifest_path is not None:
        persist_sealed_quarantine(quarantine_manifest_path, split, quarantine_config)
    return split.development_bars


def train_snapshot(
    snapshot: MarketSnapshot,
    model_configs=None,
    horizon: int = 5,
    *,
    quarantine_config: QuarantineConfig | None = None,
    quarantine_manifest_path: str | Path | None = None,
    quality_report: ResearchDataQualityReport | None = None,
    auxiliary_store: PointInTimeFeatureStore | None = None,
    auxiliary_feature_names: tuple[str, ...] | list[str] | None = None,
    auxiliary_max_age_days: int | None = None,
    auxiliary_min_coverage: float = 0.80,
) -> TrainingRun:
    return run_training_research(
        _development_bars(snapshot, quarantine_config, quarantine_manifest_path),
        _snapshot_metadata(snapshot, quality_report),
        model_configs=model_configs,
        horizon=horizon,
        auxiliary_store=auxiliary_store,
        auxiliary_feature_names=auxiliary_feature_names,
        auxiliary_max_age_days=auxiliary_max_age_days,
        auxiliary_min_coverage=auxiliary_min_coverage,
    )


def run_snapshot_factory(
    snapshot: MarketSnapshot,
    *,
    config: ResearchFactoryConfig | None = None,
    memory_path: str | None = None,
    quarantine_config: QuarantineConfig | None = None,
    quarantine_manifest_path: str | Path | None = None,
    quality_report: ResearchDataQualityReport | None = None,
    auxiliary_store: PointInTimeFeatureStore | None = None,
    auxiliary_feature_names: tuple[str, ...] | list[str] | None = None,
    auxiliary_max_age_days: int | None = None,
    auxiliary_min_coverage: float = 0.80,
) -> FactoryReport:
    """Run V2 on development data only with verified data and feature semantics."""

    memory = JsonlExperimentMemory(memory_path) if memory_path else None
    champion_store = None
    if memory_path:
        memory_file = Path(memory_path)
        champion_store = JsonChampionStore(memory_file.with_suffix(".champion.json"))

    effective_config = config or ResearchFactoryConfig()
    if memory is not None and not effective_config.population.adaptive_records:
        parents = tuple(
            memory.best(
                only_passed=True,
                limit=effective_config.population.adaptive_parent_limit,
            )
        )
        if parents:
            effective_config = replace(
                effective_config,
                population=replace(
                    effective_config.population,
                    adaptive_records=parents,
                ),
            )

    factory = ResearchFactory(
        effective_config,
        memory=memory,
        champion_store=champion_store,
    )
    return factory.run(
        _development_bars(snapshot, quarantine_config, quarantine_manifest_path),
        _snapshot_metadata(snapshot, quality_report),
        auxiliary_store=auxiliary_store,
        auxiliary_feature_names=auxiliary_feature_names,
        auxiliary_max_age_days=auxiliary_max_age_days,
        auxiliary_min_coverage=auxiliary_min_coverage,
    )


def run_snapshot_regime_specialists(
    snapshot: MarketSnapshot,
    report: FactoryReport,
    *,
    top_k_per_horizon: int = 2,
    quarantine_config: QuarantineConfig | None = None,
    quarantine_manifest_path: str | Path | None = None,
    quality_report: ResearchDataQualityReport | None = None,
    auxiliary_store: PointInTimeFeatureStore | None = None,
    auxiliary_feature_names: tuple[str, ...] | list[str] | None = None,
    auxiliary_max_age_days: int | None = None,
    auxiliary_min_coverage: float = 0.80,
) -> RegimeSpecialistDiagnostics:
    """Run regime specialists without changing the candidate feature contract."""

    return run_regime_specialist_diagnostics(
        _development_bars(snapshot, quarantine_config, quarantine_manifest_path),
        _snapshot_metadata(snapshot, quality_report),
        report.candidates,
        top_k_per_horizon=top_k_per_horizon,
        auxiliary_store=auxiliary_store,
        auxiliary_feature_names=auxiliary_feature_names,
        auxiliary_max_age_days=auxiliary_max_age_days,
        auxiliary_min_coverage=auxiliary_min_coverage,
    )


def run_snapshot_deep_diagnostics(
    snapshot: MarketSnapshot,
    report: FactoryReport,
    *,
    top_k_per_horizon: int = 1,
    train_windows: tuple[int, ...] = (126, 252, 504),
    test_periods: int = 21,
    liquidity_config: LiquidityExecutionConfig | None = None,
    capacity_levels: tuple[float, ...] = (
        10_000.0,
        25_000.0,
        50_000.0,
        100_000.0,
        250_000.0,
        500_000.0,
        1_000_000.0,
        2_500_000.0,
        5_000_000.0,
    ),
    quarantine_config: QuarantineConfig | None = None,
    quarantine_manifest_path: str | Path | None = None,
    quality_report: ResearchDataQualityReport | None = None,
    point_in_time_universe: PointInTimeUniverse | None = None,
    auxiliary_store: PointInTimeFeatureStore | None = None,
    auxiliary_feature_names: tuple[str, ...] | list[str] | None = None,
    auxiliary_max_age_days: int | None = None,
    auxiliary_min_coverage: float = 0.80,
) -> DeepResearchDiagnostics:
    """Run holdout-safe deep diagnostics with exact feature replay."""

    return run_deep_research_diagnostics(
        _development_bars(snapshot, quarantine_config, quarantine_manifest_path),
        _snapshot_metadata(snapshot, quality_report),
        report,
        top_k_per_horizon=top_k_per_horizon,
        train_windows=train_windows,
        test_periods=test_periods,
        liquidity_config=liquidity_config,
        capacity_levels=capacity_levels,
        point_in_time_universe=point_in_time_universe,
        auxiliary_store=auxiliary_store,
        auxiliary_feature_names=auxiliary_feature_names,
        auxiliary_max_age_days=auxiliary_max_age_days,
        auxiliary_min_coverage=auxiliary_min_coverage,
    )
