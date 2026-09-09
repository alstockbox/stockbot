from dataclasses import asdict

import pytest

from stockbot.research.deep_feedback import (
    build_research_cycle_id,
    make_research_cycle_manifest,
)


def test_research_cycle_manifest_binds_data_identity_but_not_execution_budget(tmp_path):
    deep_path = tmp_path / "deep-findings.jsonl"
    manifest = make_research_cycle_manifest(
        dataset_fingerprint="dataset-a",
        quarantine_start="2026-01-02",
        universe_fingerprint="universe-a",
        auxiliary_fingerprint="aux-a",
        quality_fingerprint="quality-a",
        deep_feedback_path=deep_path,
        deep_feedback_weight=0.30,
    )

    expected = build_research_cycle_id(
        dataset_fingerprint="dataset-a",
        quarantine_start="2026-01-02",
        universe_fingerprint="universe-a",
        auxiliary_fingerprint="aux-a",
        quality_fingerprint="quality-a",
    )
    assert manifest.schema_version == 1
    assert manifest.research_cycle_id == expected
    assert manifest.dataset_fingerprint == "dataset-a"
    assert manifest.quarantine_start == "2026-01-02"
    assert manifest.universe_fingerprint == "universe-a"
    assert manifest.auxiliary_fingerprint == "aux-a"
    assert manifest.quality_fingerprint == "quality-a"
    assert manifest.deep_feedback_path == str(deep_path)
    assert manifest.deep_feedback_weight == 0.30

    # Execution/search knobs are intentionally not accepted as cycle identity inputs.
    assert "workers" not in asdict(manifest)
    assert "max_candidates" not in asdict(manifest)


def test_research_cycle_id_changes_when_any_evidence_identity_changes():
    base = dict(
        dataset_fingerprint="dataset-a",
        quarantine_start="2026-01-02",
        universe_fingerprint="universe-a",
        auxiliary_fingerprint="aux-a",
        quality_fingerprint="quality-a",
    )
    baseline = build_research_cycle_id(**base)

    mutations = (
        {**base, "dataset_fingerprint": "dataset-b"},
        {**base, "quarantine_start": "2026-02-02"},
        {**base, "universe_fingerprint": "universe-b"},
        {**base, "auxiliary_fingerprint": "aux-b"},
        {**base, "quality_fingerprint": "quality-b"},
    )
    assert all(build_research_cycle_id(**payload) != baseline for payload in mutations)


def test_research_cycle_manifest_rejects_invalid_operational_feedback_settings(tmp_path):
    with pytest.raises(ValueError, match="deep_feedback_path"):
        make_research_cycle_manifest(
            dataset_fingerprint="dataset-a",
            deep_feedback_path="",
            deep_feedback_weight=0.25,
        )

    with pytest.raises(ValueError, match="deep feedback weight"):
        make_research_cycle_manifest(
            dataset_fingerprint="dataset-a",
            deep_feedback_path=tmp_path / "deep-findings.jsonl",
            deep_feedback_weight=1.01,
        )
