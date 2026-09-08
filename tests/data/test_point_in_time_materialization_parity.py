import numpy as np
import pandas as pd

from stockbot.data.point_in_time_features import (
    AuxiliaryFeatureKind,
    PointInTimeFeatureManifest,
    PointInTimeFeatureObservation,
    PointInTimeFeatureStore,
)


def _store() -> PointInTimeFeatureStore:
    rows = (
        # Global macro sequence. The late revision of the older January observation
        # must not displace the newer February observation.
        PointInTimeFeatureObservation(
            "macro", 1.0, "2026-01-01", "2026-01-01", "macro", AuxiliaryFeatureKind.MACRO
        ),
        PointInTimeFeatureObservation(
            "macro", 2.0, "2026-02-01", "2026-02-10", "macro", AuxiliaryFeatureKind.MACRO
        ),
        PointInTimeFeatureObservation(
            "macro", 1.5, "2026-01-01", "2026-03-01", "macro", AuxiliaryFeatureKind.MACRO,
            revision_id="late-old-revision",
        ),
        # Symbol-specific values override global values whenever a symbol-specific
        # observation is visible, including when that specific value becomes stale.
        PointInTimeFeatureObservation(
            "macro", 10.0, "2026-01-15", "2026-01-20", "filing", AuxiliaryFeatureKind.FUNDAMENTAL,
            symbol="AAA", revision_id="original",
        ),
        PointInTimeFeatureObservation(
            "macro", 11.0, "2026-01-15", "2026-02-20", "filing", AuxiliaryFeatureKind.FUNDAMENTAL,
            symbol="AAA", revision_id="restatement",
        ),
        PointInTimeFeatureObservation(
            "macro", 12.0, "2026-03-01", "2026-03-05", "filing", AuxiliaryFeatureKind.FUNDAMENTAL,
            symbol="AAA", revision_id="new-period",
        ),
        PointInTimeFeatureObservation(
            "sentiment", 0.2, "2026-01-05", "2026-01-05", "news", AuxiliaryFeatureKind.SENTIMENT
        ),
        PointInTimeFeatureObservation(
            "sentiment", 0.7, "2026-02-15", "2026-02-15", "news", AuxiliaryFeatureKind.SENTIMENT,
            symbol="BBB",
        ),
    )
    return PointInTimeFeatureStore(
        rows,
        PointInTimeFeatureManifest(
            source="parity",
            point_in_time=True,
            revision_aware=True,
        ),
    )


def _reference(
    store: PointInTimeFeatureStore,
    index: pd.MultiIndex,
    features: tuple[str, ...],
    *,
    max_age_days: int | None,
) -> pd.DataFrame:
    normalized, timestamps, symbols = store._normalize_index(index)
    values = np.full((len(normalized), len(features)), np.nan, dtype=float)
    for row_number, (timestamp, symbol) in enumerate(zip(timestamps, symbols)):
        for column_number, feature in enumerate(features):
            values[row_number, column_number] = store._value_asof(
                feature,
                pd.Timestamp(timestamp),
                str(symbol),
                max_age_days=max_age_days,
            )
    return pd.DataFrame(values, index=normalized, columns=list(features))


def _unsorted_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-03-10", tz="UTC"), "AAA"),
            (pd.Timestamp("2026-01-10", tz="UTC"), "BBB"),
            (pd.Timestamp("2026-02-25", tz="UTC"), "AAA"),
            (pd.Timestamp("2026-03-10", tz="UTC"), "BBB"),
            (pd.Timestamp("2026-02-25", tz="UTC"), "BBB"),
            (pd.Timestamp("2026-03-10", tz="UTC"), "AAA"),
        ],
        names=["timestamp", "symbol"],
    )


def test_vectorized_materialization_matches_reference_for_revisions_and_overrides():
    store = _store()
    index = _unsorted_index()
    features = ("macro", "sentiment")

    expected = _reference(store, index, features, max_age_days=None)
    actual = store.materialize(index, feature_names=features)
    pd.testing.assert_frame_equal(actual, expected)


def test_vectorized_materialization_matches_reference_with_max_age_and_stale_specific_override():
    store = _store()
    index = _unsorted_index()
    features = ("macro", "sentiment")

    expected = _reference(store, index, features, max_age_days=20)
    actual = store.materialize(index, feature_names=features, max_age_days=20)
    pd.testing.assert_frame_equal(actual, expected)

    # AAA has a visible symbol-specific macro observation on 2026-02-25. It is stale
    # relative to its Jan-15 observation time, so the correct result is NaN rather than
    # falling back to the still-fresh global Feb-01 macro observation.
    assert pd.isna(
        actual.loc[(pd.Timestamp("2026-02-25", tz="UTC"), "AAA"), "macro"]
    )


def test_vectorized_materialization_matches_reference_for_symbol_timestamp_level_order():
    store = _store()
    timestamp_symbol = _unsorted_index()
    symbol_timestamp = timestamp_symbol.reorder_levels([1, 0]).set_names(["symbol", "timestamp"])
    features = ("macro",)

    expected = _reference(store, symbol_timestamp, features, max_age_days=None)
    actual = store.materialize(symbol_timestamp, feature_names=features)
    pd.testing.assert_frame_equal(actual, expected)


def test_vectorized_materializer_does_not_use_per_cell_reference_path(monkeypatch):
    store = _store()

    def forbidden(*args, **kwargs):
        raise AssertionError("vectorized materialization must not call _value_asof per cell")

    monkeypatch.setattr(store, "_value_asof", forbidden)
    frame = store.materialize(_unsorted_index(), feature_names=("macro", "sentiment"))
    assert frame.shape == (6, 2)


def test_negative_max_age_is_rejected_before_materialization():
    store = _store()
    try:
        store.materialize(_unsorted_index(), feature_names=("macro",), max_age_days=-1)
    except ValueError as exc:
        assert "max_age_days" in str(exc)
    else:
        raise AssertionError("negative max_age_days must be rejected")
