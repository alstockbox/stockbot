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


class _Specialist:
    def __init__(self, score: float):
        self.score = score


class _Router:
    score = 0.66


class _Diagnostics:
    specialists = {"bull_trend": _Specialist(1.2), "neutral_chop": _Specialist(0.9)}
    router_report = _Router()
    candidate_pairs_tested = 4


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


def test_job_identity_changes_when_sealed_quarantine_boundary_changes():
    kwargs = dict(
        snapshot_id="snapshot-a",
        dataset_fingerprint="fingerprint-a",
        horizons=(1, 5, 20),
        max_candidates=160,
        max_workers=4,
        memory_path="research_memory/experiments.jsonl",
    )
    first = make_job_manifest(**kwargs, quarantine_start="2026-01-01")
    second = make_job_manifest(**kwargs, quarantine_start="2026-03-01")
    assert first.job_id != second.job_id
    assert first.quarantine_start == "2026-01-01"


def test_run_summary_and_atomic_json_writer(tmp_path):
    manifest = make_job_manifest(
        snapshot_id="snapshot-a",
        dataset_fingerprint="fingerprint-a",
        horizons=(5,),
        max_candidates=10,
        max_workers=2,
        memory_path=None,
    )
    summary = make_run_summary(manifest.job_id, _Report(), _Diagnostics())
    target = tmp_path / "summary.json"
    write_json_record(target, summary)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["job_id"] == manifest.job_id
    assert payload["experiments_run"] == 120
    assert payload["ensemble_score"] == 0.72
    assert payload["champion_experiment_id"] == "candidate-1"
    assert payload["specialist_router_score"] == 0.66
    assert payload["specialist_scores"]["bull_trend"] == 1.2
    assert payload["specialist_candidate_pairs_tested"] == 4
    assert payload["schema_version"] == 2
