from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from stockbot.ml.models import ModelConfig


@dataclass(frozen=True)
class ModelPopulationConfig:
    """Deterministic search space for challenger generation."""

    seeds: tuple[int, ...] = (7, 19)
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0)
    elastic_alphas: tuple[float, ...] = (0.0003, 0.001, 0.003)
    elastic_l1_ratios: tuple[float, ...] = (0.1, 0.25, 0.5, 0.8)
    tree_depths: tuple[int, ...] = (4, 8, 12)
    tree_leaf_sizes: tuple[int, ...] = (2, 4, 8)
    tree_estimators: tuple[int, ...] = (120, 240)
    hist_learning_rates: tuple[float, ...] = (0.025, 0.05, 0.1)
    hist_leaf_nodes: tuple[int, ...] = (7, 15, 31)
    hist_l2: tuple[float, ...] = (0.0, 0.1, 1.0)
    max_candidates: int = 160

    def __post_init__(self) -> None:
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if not self.seeds:
            raise ValueError("at least one seed is required")


def _families(config: ModelPopulationConfig) -> list[list[ModelConfig]]:
    ridge = [
        ModelConfig("ridge", {"alpha": alpha}, seed=seed)
        for seed, alpha in product(config.seeds, config.ridge_alphas)
    ]
    elastic = [
        ModelConfig("elastic_net", {"alpha": alpha, "l1_ratio": ratio}, seed=seed)
        for seed, alpha, ratio in product(config.seeds, config.elastic_alphas, config.elastic_l1_ratios)
    ]
    extra_trees = [
        ModelConfig(
            "extra_trees",
            {"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf},
            seed=seed,
        )
        for seed, n, depth, leaf in product(
            config.seeds, config.tree_estimators, config.tree_depths, config.tree_leaf_sizes
        )
    ]
    random_forest = [
        ModelConfig(
            "random_forest",
            {"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf},
            seed=seed,
        )
        for seed, n, depth, leaf in product(
            config.seeds, config.tree_estimators, config.tree_depths, config.tree_leaf_sizes
        )
    ]
    hist_gb = [
        ModelConfig(
            "hist_gb",
            {
                "learning_rate": rate,
                "max_leaf_nodes": nodes,
                "l2_regularization": l2,
            },
            seed=seed,
        )
        for seed, rate, nodes, l2 in product(
            config.seeds,
            config.hist_learning_rates,
            config.hist_leaf_nodes,
            config.hist_l2,
        )
    ]
    return [ridge, elastic, extra_trees, random_forest, hist_gb]


def generate_model_population(config: ModelPopulationConfig | None = None) -> list[ModelConfig]:
    """Create a broad but balanced challenger population.

    Candidates are interleaved by model family rather than taking one giant grid in
    family order. This guarantees that a capped experiment budget still explores
    linear, bagged-tree and boosting families.
    """

    cfg = config or ModelPopulationConfig()
    families = _families(cfg)
    population: list[ModelConfig] = []
    cursor = 0
    while len(population) < cfg.max_candidates:
        added = False
        for family in families:
            if cursor < len(family):
                population.append(family[cursor])
                added = True
                if len(population) >= cfg.max_candidates:
                    break
        if not added:
            break
        cursor += 1
    return population
