from __future__ import annotations

import pandas as pd

from stockbot.ml.challenger import MLChallenger
from stockbot.ml.validation import walk_forward_splits


def walk_forward_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: int,
    test_size: int,
    embargo: int = 0,
    model_type: str = "ridge",
    seed: int = 7,
) -> pd.Series:
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same ordered index")
    predictions = pd.Series(float("nan"), index=X.index, dtype=float, name="prediction")
    for train_idx, test_idx in walk_forward_splits(
        n_samples=len(X),
        train_size=train_size,
        test_size=test_size,
        embargo=embargo,
    ):
        model = MLChallenger(model_type=model_type, seed=seed)
        model.fit(X.iloc[train_idx].to_numpy(), y.iloc[train_idx].to_numpy())
        predictions.iloc[test_idx] = model.predict_score(X.iloc[test_idx].to_numpy())
    return predictions
