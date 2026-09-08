from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stockbot.data.market_schema import validate_canonical_bars
from stockbot.data.schemas import DatasetMetadata
from stockbot.data.snapshots import MarketSnapshot
from stockbot.research.champion import JsonChampionStore
from stockbot.research.factory import FactoryReport, ResearchFactory, ResearchFactoryConfig
from stockbot.research.memory import JsonlExperimentMemory
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


def _snapshot_metadata(snapshot: MarketSnapshot) -> DatasetMetadata:
    return DatasetMetadata(
        name=snapshot.snapshot_id,
        source=snapshot.manifest.provider,
        grade=snapshot.manifest.grade,
        version=snapshot.manifest.schema_version,
        created_at=snapshot.manifest.created_at,
    )


def train_snapshot(
    snapshot: MarketSnapshot,
    model_configs=None,
    horizon: int = 5,
) -> TrainingRun:
    return run_training_research(
        prepare_training_bars(snapshot.bars),
        _snapshot_metadata(snapshot),
        model_configs=model_configs,
        horizon=horizon,
    )


def run_snapshot_factory(
    snapshot: MarketSnapshot,
    *,
    config: ResearchFactoryConfig | None = None,
    memory_path: str | None = None,
) -> FactoryReport:
    """Run the V2 multi-horizon research factory on an immutable market snapshot."""

    memory = JsonlExperimentMemory(memory_path) if memory_path else None
    champion_store = None
    if memory_path:
        memory_file = Path(memory_path)
        champion_store = JsonChampionStore(memory_file.with_suffix(".champion.json"))
    factory = ResearchFactory(
        config,
        memory=memory,
        champion_store=champion_store,
    )
    return factory.run(
        prepare_training_bars(snapshot.bars),
        _snapshot_metadata(snapshot),
    )


def run_snapshot_regime_specialists(
    snapshot: MarketSnapshot,
    report: FactoryReport,
    *,
    top_k_per_horizon: int = 2,
) -> RegimeSpecialistDiagnostics:
    """Run the optional V2.1 regime-specialist diagnostics on a finished factory report."""

    return run_regime_specialist_diagnostics(
        prepare_training_bars(snapshot.bars),
        _snapshot_metadata(snapshot),
        report.candidates,
        top_k_per_horizon=top_k_per_horizon,
    )
