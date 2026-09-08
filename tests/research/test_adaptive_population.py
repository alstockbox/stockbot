from stockbot.ml.models import ModelConfig
from stockbot.research.memory import make_record
from stockbot.research.population import ModelPopulationConfig, generate_model_population


def test_population_evolves_around_historical_winner_without_losing_budget():
    parent = make_record(
        ModelConfig("ridge", {"alpha": 100.0}, seed=7),
        dataset_fingerprint="old-dataset",
        horizon=5,
        factory_score=10.0,
        base_score=8.0,
        robustness=0.8,
        oos_coverage=0.7,
        metrics={"cagr": 0.2},
        passed_gates=True,
    )
    config = ModelPopulationConfig(
        max_candidates=10,
        seeds=(7,),
        adaptive_records=(parent,),
        adaptive_fraction=0.4,
        adaptive_parent_limit=1,
        adaptive_mutations_per_parent=4,
    )

    population = generate_model_population(config)
    ridge_alphas = {
        float(model.params["alpha"])
        for model in population
        if model.name == "ridge" and "alpha" in model.params
    }

    assert len(population) == 10
    assert len({(model.name, tuple(sorted(model.params.items())), model.seed) for model in population}) == 10
    assert any(alpha in ridge_alphas for alpha in (50.0, 80.0, 125.0, 200.0))
    assert {model.name for model in population}.issuperset({"ridge", "elastic_net", "extra_trees"})


def test_population_stays_broad_when_no_research_memory_exists():
    config = ModelPopulationConfig(max_candidates=10, seeds=(7,), adaptive_records=())
    population = generate_model_population(config)
    assert len(population) == 10
    assert {model.name for model in population} == {
        "ridge",
        "elastic_net",
        "extra_trees",
        "random_forest",
        "hist_gb",
    }
