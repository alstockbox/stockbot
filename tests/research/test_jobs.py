import json

from stockbot.research.jobs import make_job_manifest, make_run_summary, write_json_record


class _Candidate:
    experiment_id = "candidate-1"


class _Champion:
    experiment_id = "champion-1"


class _Ensemble:
    score = 0.72


class _Report:
    experiments_run = 120
    candidates_passed = 4
    holdout_evaluated = 8
    promotion_occurred = True
    champion_candidate = _Candidate()
    active_champion = _Champion()
    ensemble_report = _Ensemble()


def test_job_manifest_is_deterministic_for_same_research_inputs():
    kwargs = dict(
        snapshot_id="snapshot-a",
        dataset_fingerprint="fingerprint-a",
        horizons=(1, 5, 20),
        max_candidates=160,
        max_workers=4,
        memory_path="research_memory/experiments.jsonl",
    )
    first = make_job_manifest(**kwargs)
    second = make_job_manifest(**kwargs)
    assert first.job_id == second.job_id
    assert first.horizons == (1, 5, 20)


def test_run_summary_and_atomic_json_writer(tmp_path):
    manifest = make_job_manifest(
        snapshot_id="snapshot-a",
        dataset_fingerprint="fingerprint-a",
        horizons=(5,),
        max_candidates=10,
        max_workers=2,
        memory_path=None,
    )
    summary = make_run_summary(manifest.job_id, _Report())
    target = tmp_path / "summary.json"
    write_json_record(target, summary)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["job_id"] == manifest.job_id
    assert payload["experiments_run"] == 120
    assert payload["ensemble_score"] == 0.72
    assert payload["champion_experiment_id"] == "candidate-1"
