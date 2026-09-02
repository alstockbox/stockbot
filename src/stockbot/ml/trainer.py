from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockbot.data.schemas import DatasetMetadata
from stockbot.ml.artifacts import ExperimentArtifact, dataset_fingerprint
from stockbot.ml.models import ModelConfig, build_model
from stockbot.ml.purged_cv import PurgedWalkForwardSplitter


@dataclass(frozen=True)
class OOSResult:
    predictions: pd.Series
    artifact: ExperimentArtifact


def train_oos(features: pd.DataFrame, labels: pd.Series, splitter: PurgedWalkForwardSplitter, model_config: ModelConfig, metadata: DatasetMetadata, label_name: str = "target") -> OOSResult:
    if not features.index.equals(labels.index):
        labels = labels.reindex(features.index)
    predictions = pd.Series(np.nan, index=features.index, dtype=float, name=f"pred_{label_name}")
    fold_count = 0
    for train_idx, test_idx in splitter.split(features.index):
        X_train, y_train, X_test = features.iloc[train_idx], labels.iloc[train_idx], features.iloc[test_idx]
        valid_train = X_train.notna().all(axis=1) & y_train.notna() & np.isfinite(y_train.astype(float))
        valid_test = X_test.notna().all(axis=1)
        if int(valid_train.sum()) < 10 or int(valid_test.sum()) == 0:
            continue
        model = build_model(model_config)
        model.fit(X_train.loc[valid_train].to_numpy(dtype=float), y_train.loc[valid_train].to_numpy(dtype=float))
        pred = np.asarray(model.predict(X_test.loc[valid_test].to_numpy(dtype=float)), dtype=float)
        if not np.all(np.isfinite(pred)):
            raise ValueError("model produced non-finite OOS predictions")
        predictions.loc[X_test.loc[valid_test].index] = pred
        fold_count += 1
    artifact = ExperimentArtifact(dataset_fingerprint(features, metadata), tuple(str(c) for c in features.columns), label_name, model_config.name, dict(model_config.params), int(model_config.seed), fold_count, float(predictions.notna().mean()), {})
    return OOSResult(predictions=predictions, artifact=artifact)
