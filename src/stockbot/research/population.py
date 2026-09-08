from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from stockbot.ml.models import ModelConfig
from stockbot.research.adaptive_population import AdaptivePopulationConfig, generate_adaptive_population
from stockbot.research.memory import ExperimentRecord


@dataclass(frozen=True)
class ModelPopulationConfig:
    """Deterministic search space for broad + adaptive challenger generation."""

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
    adaptive_records: tuple[ExperimentRecord, ...] = ()
    adaptive_fraction: float = 0.35
    adaptive_parent_limit: int = 12
    adaptive_mutations_per_parent: int = 4

    def __post_init__(self) -> None:
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        AdaptivePopulationConfig(
            adaptive_fraction=self.adaptive_fraction,
            parent_limit=self.adaptive_parent_limit,
            mutations_per_parent=self.adaptive_mutations_per_parent,
        )


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


def _broad_population(cfg: ModelPopulationConfig) -> list[ModelConfig]:
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


def generate_model_population(config: ModelPopulationConfig | None = None) -> list[ModelConfig]:
    """Create a balanced population, optionally evolved from prior research memory.

    Broad grid candidates remain interleaved by model family. When historical records
    are supplied, a bounded fraction of the budget is replaced by local mutations of
    the strongest prior experiments, preserving exploration while accelerating search
    around regions that previously showed promising OOS evidence.
    """

    cfg = config or ModelPopulationConfig()
    broad = _broad_population(cfg)
    if not cfg.adaptive_records or cfg.adaptive_fraction <= 0.0:
        return broad

    return generate_adaptive_population(
        broad,
        cfg.adaptive_records,
        max_candidates=cfg.max_candidates,
        config=AdaptivePopulationConfig(
            adaptive_fraction=cfg.adaptive_fraction,
            parent_limit=cfg.adaptive_parent_limit,
            mutations_per_parent=cfg.adaptive_mutations_per_parent,
        ),
    )
