import pandas as pd

from stockbot.data.ingestion import (
    point_in_time_feature_store_from_frame,
    point_in_time_universe_from_frame,
)
from stockbot.data.point_in_time_features import PointInTimeFeatureManifest
from stockbot.data.universe import UniverseManifest


def test_feature_ingestion_requires_explicit_available_time():
    frame = pd.DataFrame(
        [
            {
                "feature_name": "eps_ttm",
                "value": 1.2,
                "observation_time": "2025-12-31",
                "source": "filings",
                "kind": "fundamental",
                "symbol": "AAA",
            }
        ]
    )
    try:
        point_in_time_feature_store_from_frame(
            frame,
            PointInTimeFeatureManifest(
                source="synthetic",
                point_in_time=True,
                revision_aware=True,
            ),
        )
    except ValueError as exc:
        assert "available_time" in str(exc)
    else:
        raise AssertionError("expected missing available_time to fail closed")


def test_feature_ingestion_rejects_unknown_columns_instead_of_guessing_semantics():
    frame = pd.DataFrame(
        [
            {
                "feature_name": "eps_ttm",
                "value": 1.2,
                "observation_time": "2025-12-31",
                "available_time": "2026-02-15",
                "source": "filings",
                "kind": "fundamental",
                "symbol": "AAA",
                "published_maybe": "2026-02-10",
            }
        ]
    )
    try:
        point_in_time_feature_store_from_frame(
            frame,
            PointInTimeFeatureManifest(
                source="synthetic",
                point_in_time=True,
                revision_aware=True,
            ),
        )
    except ValueError as exc:
        assert "unsupported columns" in str(exc)
    else:
        raise AssertionError("expected ambiguous provider column to be rejected")


def test_feature_ingestion_normalizes_kind_symbol_and_revision():
    frame = pd.DataFrame(
        [
            {
                "feature_name": "eps_ttm",
                "value": 1.2,
                "observation_time": "2025-12-31",
                "available_time": "2026-02-15",
                "source": "filings",
                "kind": "FUNDAMENTAL",
                "symbol": " aaa ",
                "revision_id": "original",
            }
        ]
    )
    store = point_in_time_feature_store_from_frame(
        frame,
        PointInTimeFeatureManifest(
            source="synthetic",
            point_in_time=True,
            revision_aware=True,
        ),
    )
    assert store.observations[0].kind.value == "fundamental"
    assert store.to_payload()["observations"][0]["symbol"] == "AAA"


def test_universe_ingestion_parses_delisted_boolean_strictly():
    frame = pd.DataFrame(
        [
            {
                "symbol": "OLD",
                "effective_from": "2024-01-01",
                "effective_to": "2025-06-01",
                "exchange": "XSTO",
                "sector": "Industrials",
                "industry": "Machinery",
                "delisted": "yes",
            }
        ]
    )
    universe = point_in_time_universe_from_frame(
        frame,
        UniverseManifest(
            source="synthetic",
            survivorship_bias_controlled=True,
            includes_delisted_securities=True,
            point_in_time_membership=True,
        ),
    )
    assert universe.memberships[0].delisted is True


def test_universe_ingestion_rejects_ambiguous_boolean():
    frame = pd.DataFrame(
        [
            {
                "symbol": "OLD",
                "effective_from": "2024-01-01",
                "delisted": "probably",
            }
        ]
    )
    try:
        point_in_time_universe_from_frame(
            frame,
            UniverseManifest(
                source="synthetic",
                survivorship_bias_controlled=True,
                includes_delisted_securities=True,
                point_in_time_membership=True,
            ),
        )
    except ValueError as exc:
        assert "boolean" in str(exc)
    else:
        raise AssertionError("expected ambiguous delisted value to be rejected")
