from __future__ import annotations

import numpy as np
import pandas as pd

from stockbot.data.market_schema import validate_canonical_bars
from stockbot.data.schemas import DatasetMetadata
from stockbot.data.snapshots import MarketSnapshot
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


def train_snapshot(
    snapshot: MarketSnapshot,
    model_configs=None,
    horizon: int = 5,
) -> TrainingRun:
    metadata = DatasetMetadata(
        name=snapshot.snapshot_id,
        source=snapshot.manifest.provider,
        grade=snapshot.manifest.grade,
        version=snapshot.manifest.schema_version,
        created_at=snapshot.manifest.created_at,
    )
    return run_training_research(
        prepare_training_bars(snapshot.bars),
        metadata,
        model_configs=model_configs,
        horizon=horizon,
    )
