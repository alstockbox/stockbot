import json

from stockbot.data.point_in_time_features import (
    AuxiliaryFeatureKind,
    PointInTimeFeatureManifest,
    PointInTimeFeatureObservation,
    PointInTimeFeatureStore,
)
from stockbot.data.research_inputs import (
    load_point_in_time_feature_store,
    load_point_in_time_universe,
    write_point_in_time_feature_store,
    write_point_in_time_universe,
)
from stockbot.data.universe import PointInTimeUniverse, UniverseManifest, UniverseMembership


def _feature_store():
    return PointInTimeFeatureStore(
        (
            PointInTimeFeatureObservation(
                feature_name="eps_ttm",
                value=1.25,
                observation_time="2025-12-31",
                available_time="2026-02-15",
                source="filings",
                kind=AuxiliaryFeatureKind.FUNDAMENTAL,
                symbol="AAA",
                revision_id="original",
            ),
            PointInTimeFeatureObservation(
                feature_name="policy_rate",
                value=2.50,
                observation_time="2026-01-28",
                available_time="2026-01-28",
                source="central-bank",
                kind=AuxiliaryFeatureKind.MACRO,
            ),
        ),
        PointInTimeFeatureManifest(
            source="synthetic",
            point_in_time=True,
            revision_aware=True,
        ),
    )


def _universe():
    return PointInTimeUniverse(
        (
            UniverseMembership(
                symbol="AAA",
                effective_from="2024-01-01",
                sector="Technology",
                industry="Software",
            ),
            UniverseMembership(
                symbol="OLD",
                effective_from="2024-01-01",
                effective_to="2025-06-01",
                sector="Industrials",
                industry="Machinery",
                delisted=True,
            ),
        ),
        UniverseManifest(
            source="synthetic",
            survivorship_bias_controlled=True,
            includes_delisted_securities=True,
            point_in_time_membership=True,
        ),
    )


def test_feature_store_round_trip_preserves_fingerprint(tmp_path):
    store = _feature_store()
    path = tmp_path / "features.json"
    write_point_in_time_feature_store(path, store)
    loaded = load_point_in_time_feature_store(path)
    assert loaded.fingerprint == store.fingerprint
    assert loaded.to_payload() == store.to_payload()


def test_universe_round_trip_preserves_fingerprint(tmp_path):
    universe = _universe()
    path = tmp_path / "universe.json"
    write_point_in_time_universe(path, universe)
    loaded = load_point_in_time_universe(path)
    assert loaded.fingerprint == universe.fingerprint
    assert loaded.to_payload() == universe.to_payload()


def test_feature_store_fingerprint_is_order_independent():
    store = _feature_store()
    reversed_store = PointInTimeFeatureStore(
        tuple(reversed(store.observations)),
        store.manifest,
    )
    assert reversed_store.fingerprint == store.fingerprint


def test_tampered_research_input_is_rejected(tmp_path):
    path = tmp_path / "features.json"
    write_point_in_time_feature_store(path, _feature_store())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["payload"]["observations"][0]["value_hex"] = float(999.0).hex()
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_point_in_time_feature_store(path)
    except ValueError as exc:
        assert "fingerprint mismatch" in str(exc)
    else:
        raise AssertionError("expected tampered research input to be rejected")
