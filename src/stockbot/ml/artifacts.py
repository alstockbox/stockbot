from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping

import pandas as pd

from stockbot.data.schemas import DatasetMetadata


@dataclass(frozen=True)
class ExperimentArtifact:
    dataset_fingerprint: str
    feature_names: tuple[str, ...]
    label_name: str
    model_name: str
    model_params: Mapping[str, Any]
    seed: int
    fold_count: int
    oos_coverage: float
    metrics: Mapping[str, float] = field(default_factory=dict)


def dataset_fingerprint(frame: pd.DataFrame, metadata: DatasetMetadata) -> str:
    h = hashlib.sha256()
    stable_meta = {"name": metadata.name, "source": metadata.source, "grade": metadata.grade.value, "version": metadata.version}
    h.update(json.dumps(stable_meta, sort_keys=True, separators=(",", ":")).encode())
    h.update(json.dumps([str(c) for c in frame.columns], separators=(",", ":")).encode())
    h.update(json.dumps([str(t) for t in frame.dtypes], separators=(",", ":")).encode())
    h.update(pd.util.hash_pandas_object(frame, index=True, categorize=False).to_numpy().tobytes())
    return h.hexdigest()
