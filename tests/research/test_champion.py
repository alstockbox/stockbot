from stockbot.research.champion import JsonChampionStore, make_champion_state


def test_champion_store_round_trip_and_atomic_replace(tmp_path):
    store = JsonChampionStore(tmp_path / "champion.json")
    first = make_champion_state(
        experiment_id="abc",
        horizon=5,
        model_name="ridge",
        model_params={"alpha": 1.0},
        seed=7,
        dataset_fingerprint="dataset-a",
        factory_score=1.2,
        promotion_score=1.1,
        holdout_score=0.9,
    )
    second = make_champion_state(
        experiment_id="def",
        horizon=20,
        model_name="hist_gb",
        model_params={"learning_rate": 0.05},
        seed=19,
        dataset_fingerprint="dataset-b",
        factory_score=1.5,
        promotion_score=1.4,
        holdout_score=1.2,
    )

    assert store.load() is None
    store.save(first)
    assert store.load() == first
    store.save(second)
    assert store.load() == second
