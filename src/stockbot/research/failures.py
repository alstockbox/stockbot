from __future__ import annotations

import numpy as np
import pandas as pd


def mine_hard_negatives(
    predictions: pd.Series,
    labels: pd.Series,
    *,
    confidence_quantile: float = 0.80,
    limit: int = 500,
) -> pd.DataFrame:
    """Return high-confidence directional mistakes for targeted retraining.

    This is deliberately model-agnostic. Predictions and labels are aligned by index;
    examples where the model had a large absolute prediction but the realized target
    moved in the opposite direction receive the highest severity.
    """

    if not 0.0 < confidence_quantile < 1.0:
        raise ValueError("confidence_quantile must be in (0,1)")
    if limit <= 0:
        raise ValueError("limit must be positive")

    frame = pd.concat(
        [predictions.rename("prediction"), labels.rename("label")],
        axis=1,
        join="inner",
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if frame.empty:
        return pd.DataFrame(columns=["prediction", "label", "confidence", "error", "severity"])

    frame["confidence"] = frame["prediction"].abs()
    threshold = float(frame["confidence"].quantile(confidence_quantile))
    confident = frame.loc[frame["confidence"] >= threshold].copy()
    directional_error = np.sign(confident["prediction"]) != np.sign(confident["label"])
    confident = confident.loc[directional_error].copy()

    if confident.empty:
        return pd.DataFrame(columns=["prediction", "label", "confidence", "error", "severity"])

    confident["error"] = (confident["label"] - confident["prediction"]).abs()
    confident["severity"] = confident["confidence"] * confident["error"]
    confident = confident.sort_values(["severity", "confidence"], ascending=False)
    return confident.head(limit)
