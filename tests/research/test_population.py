from stockbot.research.population import ModelPopulationConfig, generate_model_population


def test_population_is_deterministic_and_balanced():
    config = ModelPopulationConfig(max_candidates=15, seeds=(7,))
    first = generate_model_population(config)
    second = generate_model_population(config)

    assert first == second
    assert len(first) == 15
    assert {model.name for model in first} == {
        "ridge",
        "elastic_net",
        "extra_trees",
        "random_forest",
        "hist_gb",
    }


def test_population_respects_budget():
    population = generate_model_population(ModelPopulationConfig(max_candidates=7))
    assert len(population) == 7
