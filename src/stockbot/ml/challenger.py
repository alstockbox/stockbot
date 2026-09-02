from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class MLChallenger:
    def __init__(self, model_type: str = "ridge", seed: int = 7) -> None:
        self.model_type = model_type
        self.seed = int(seed)
        self.model = self._build_model()

    def _build_model(self):
        if self.model_type == "ridge":
            return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        if self.model_type in {"hist_gb", "hist_gradient_boosting"}:
            return HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=15,
                l2_regularization=0.1,
                random_state=self.seed,
            )
        raise ValueError(f"unsupported model_type: {self.model_type}")

    def fit(self, X, y) -> "MLChallenger":
        self.model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict_score(self, X) -> np.ndarray:
        values = np.asarray(self.model.predict(np.asarray(X, dtype=float)), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("model produced non-finite predictions")
        return values
