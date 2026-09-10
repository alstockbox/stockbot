from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from stockbot.ml.models import ModelConfig
from stockbot.research.memory import ExperimentRecord


@dataclass(frozen=True)
class AdaptivePopulationConfig:
    adaptive_fraction: float = 0.35
    parent_limit: int = 12
    mutations_per_parent: int = 4

    def __post_init__(self) -> None:
        if not 0.0 <= self.adaptive_fraction <= 0.80:
            raise ValueError("adaptive_fraction must be in [0,0.8]")
        if self.parent_limit <= 0:
            raise ValueError("parent_limit must be positive")
        if self.mutations_per_parent <= 0:
            raise ValueError("mutations_per_parent must be positive")


def _key(config: ModelConfig) -> tuple[str, tuple[tuple[str, object], ...], int]:
    return config.name, tuple(sorted(dict(config.params).items())), int(config.seed)


def _positive(value: float, minimum: float = 1e-8) -> float:
    return max(minimum, float(value))


def _mutations(parent: ExperimentRecord) -> list[ModelConfig]:
    name = parent.model_name
    params = dict(parent.model_params)
    seed = int(parent.seed)
    output: list[ModelConfig] = []

    if name == "ridge":
        alpha = float(params.get("alpha", 1.0))
        for factor in (0.5, 0.8, 1.25, 2.0):
            output.append(ModelConfig(name, {**params, "alpha": _positive(alpha * factor)}, seed=seed))

    elif name == "elastic_net":
        alpha = float(params.get("alpha", 0.001))
        ratio = float(params.get("l1_ratio", 0.25))
        variants = (
            (0.5, -0.10),
            (0.8, 0.10),
            (1.25, -0.20),
            (2.0, 0.20),
        )
        for factor, delta in variants:
            output.append(
                ModelConfig(
                    name,
                    {
                        **params,
                        "alpha": _positive(alpha * factor),
                        "l1_ratio": min(0.99, max(0.01, ratio + delta)),
                    },
                    seed=seed,
                )
            )

    elif name in {"extra_trees", "random_forest"}:
        n = int(params.get("n_estimators", 200))
        depth = params.get("max_depth", 8)
        depth = 8 if depth is None else int(depth)
        leaf = int(params.get("min_samples_leaf", 4))
        variants = (
            (max(50, int(n * 0.75)), max(2, depth - 2), max(1, leaf - 1)),
            (max(50, int(n * 1.25)), depth + 2, leaf),
            (max(50, int(n * 1.5)), max(2, depth - 1), leaf + 1),
            (max(50, n), depth + 4, max(1, leaf - 2)),
        )
        for estimators, variant_depth, variant_leaf in variants:
            output.append(
                ModelConfig(
                    name,
                    {
                        **params,
                        "n_estimators": estimators,
                        "max_depth": variant_depth,
                        "min_samples_leaf": variant_leaf,
                    },
                    seed=seed,
                )
            )

    elif name == "hist_gb":
        rate = float(params.get("learning_rate", 0.05))
        nodes = int(params.get("max_leaf_nodes", 15))
        l2 = float(params.get("l2_regularization", 0.1))
        variants = (
            (0.6, max(3, nodes // 2), 0.5),
            (0.8, max(3, nodes - 4), 1.5),
            (1.25, nodes + 4, 0.8),
            (1.6, min(127, nodes * 2 + 1), 2.0),
        )
        for rate_factor, variant_nodes, l2_factor in variants:
            output.append(
                ModelConfig(
                    name,
                    {
                        **params,
                        "learning_rate": min(0.5, _positive(rate * rate_factor)),
                        "max_leaf_nodes": variant_nodes,
                        "l2_regularization": max(0.0, l2 * l2_factor),
                    },
                    seed=seed,
                )
            )

    return output


def generate_adaptive_population(
    base_population: list[ModelConfig],
    historical_records: Iterable[ExperimentRecord],
    *,
    max_candidates: int,
    config: AdaptivePopulationConfig | None = None,
) -> list[ModelConfig]:
    """Mix broad exploration with mutations around historically strong challengers."""

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    cfg = config or AdaptivePopulationConfig()
    records = sorted(
        list(historical_records),
        key=lambda row: float(row.factory_score),
        reverse=True,
    )[: cfg.parent_limit]

    adaptive_budget = min(
        max_candidates,
        int(round(max_candidates * cfg.adaptive_fraction)),
    )
    base_budget = max_candidates - adaptive_budget

    output: list[ModelConfig] = []
    seen: set[tuple[str, tuple[tuple[str, object], ...], int]] = set()

    def add(model: ModelConfig) -> bool:
        key = _key(model)
        if key in seen or len(output) >= max_candidates:
            return False
        seen.add(key)
        output.append(model)
        return True

    for model in base_population[:base_budget]:
        add(model)

    mutations_added = 0
    for parent in records:
        for model in _mutations(parent)[: cfg.mutations_per_parent]:
            if mutations_added >= adaptive_budget:
                break
            if add(model):
                mutations_added += 1
        if mutations_added >= adaptive_budget:
            break

    # If memory was sparse or mutations collided, fill remaining capacity with broad search.
    for model in base_population:
        if len(output) >= max_candidates:
            break
        add(model)

    return output
