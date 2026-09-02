from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 7


def build_model(config: ModelConfig):
    params = dict(config.params)
    name = config.name
    if name == "ridge":
        alpha = float(params.pop("alpha", 1.0))
        if params:
            raise ValueError(f"unsupported ridge params: {sorted(params)}")
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if name == "elastic_net":
        defaults = {"alpha": 0.001, "l1_ratio": 0.25, "max_iter": 5000, "random_state": config.seed}; defaults.update(params)
        return make_pipeline(StandardScaler(), ElasticNet(**defaults))
    if name == "extra_trees":
        defaults = {"n_estimators": 120, "max_depth": 8, "min_samples_leaf": 4, "random_state": config.seed, "n_jobs": 1}; defaults.update(params)
        return ExtraTreesRegressor(**defaults)
    if name == "random_forest":
        defaults = {"n_estimators": 120, "max_depth": 8, "min_samples_leaf": 4, "random_state": config.seed, "n_jobs": 1}; defaults.update(params)
        return RandomForestRegressor(**defaults)
    if name in {"hist_gb", "hist_gradient_boosting"}:
        defaults = {"learning_rate": 0.05, "max_iter": 150, "max_leaf_nodes": 15, "l2_regularization": 0.1, "random_state": config.seed}; defaults.update(params)
        return HistGradientBoostingRegressor(**defaults)
    raise ValueError(f"unsupported model_type: {name}")
