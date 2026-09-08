from stockbot.ml.models import ModelConfig
from stockbot.research.memory import JsonlExperimentMemory, make_record


def test_experiment_memory_round_trip_and_best(tmp_path):
    memory = JsonlExperimentMemory(tmp_path / "research.jsonl")
    model = ModelConfig("ridge", {"alpha": 1.0}, seed=7)

    weak = make_record(
        model,
        dataset_fingerprint="dataset-a",
        horizon=5,
        factory_score=0.5,
        base_score=0.4,
        robustness=0.7,
        oos_coverage=0.8,
        metrics={"cagr": 0.1},
        passed_gates=True,
    )
    strong = make_record(
        ModelConfig("ridge", {"alpha": 10.0}, seed=7),
        dataset_fingerprint="dataset-a",
        horizon=5,
        factory_score=1.5,
        base_score=1.0,
        robustness=0.8,
        oos_coverage=0.9,
        metrics={"cagr": 0.2},
        passed_gates=True,
    )

    memory.append(weak)
    memory.append(strong)

    rows = memory.records()
    assert len(rows) == 2
    assert memory.seen(weak.experiment_id)
    assert memory.best(limit=1)[0].experiment_id == strong.experiment_id
