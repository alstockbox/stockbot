import pandas as pd

from stockbot.data.point_in_time_features import (
    AuxiliaryFeatureKind,
    PointInTimeFeatureManifest,
    PointInTimeFeatureObservation,
    PointInTimeFeatureStore,
)


def _index():
    return pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-02-01", tz="UTC"), "AAA"),
            (pd.Timestamp("2026-02-16", tz="UTC"), "AAA"),
            (pd.Timestamp("2026-03-02", tz="UTC"), "AAA"),
            (pd.Timestamp("2026-03-02", tz="UTC"), "BBB"),
        ],
        names=["timestamp", "symbol"],
    )


def _store(extra=()):
    observations = (
        PointInTimeFeatureObservation(
            feature_name="eps_ttm",
            value=1.20,
            observation_time="2025-12-31",
            available_time="2026-02-15",
            source="filings",
            kind=AuxiliaryFeatureKind.FUNDAMENTAL,
            symbol="AAA",
            revision_id="original",
        ),
        PointInTimeFeatureObservation(
            feature_name="eps_ttm",
            value=1.30,
            observation_time="2025-12-31",
            available_time="2026-03-01",
            source="filings",
            kind=AuxiliaryFeatureKind.FUNDAMENTAL,
            symbol="AAA",
            revision_id="restatement-1",
        ),
        PointInTimeFeatureObservation(
            feature_name="eps_ttm",
            value=2.00,
            observation_time="2025-12-31",
            available_time="2026-02-10",
            source="filings",
            kind=AuxiliaryFeatureKind.FUNDAMENTAL,
            symbol="BBB",
            revision_id="original",
        ),
        PointInTimeFeatureObservation(
            feature_name="policy_rate",
            value=2.50,
            observation_time="2026-01-28",
            available_time="2026-01-28",
            source="central-bank",
            kind=AuxiliaryFeatureKind.MACRO,
            symbol=None,
        ),
    ) + tuple(extra)
    return PointInTimeFeatureStore(
        observations,
        PointInTimeFeatureManifest(
            source="synthetic",
            point_in_time=True,
            revision_aware=True,
        ),
    )


def test_materialization_never_uses_filing_before_available_time():
    frame = _store().materialize(_index(), feature_names=("eps_ttm", "policy_rate"))

    assert pd.isna(frame.loc[(pd.Timestamp("2026-02-01", tz="UTC"), "AAA"), "eps_ttm"])
    assert frame.loc[(pd.Timestamp("2026-02-16", tz="UTC"), "AAA"), "eps_ttm"] == 1.20
    assert frame.loc[(pd.Timestamp("2026-03-02", tz="UTC"), "AAA"), "eps_ttm"] == 1.30
    assert frame.loc[(pd.Timestamp("2026-03-02", tz="UTC"), "BBB"), "eps_ttm"] == 2.00
    assert (frame["policy_rate"] == 2.50).all()


def test_later_restatement_does_not_rewrite_pre_revision_history():
    store = _store()
    before = store.materialize(_index(), feature_names=("eps_ttm",))
    assert before.loc[(pd.Timestamp("2026-02-16", tz="UTC"), "AAA"), "eps_ttm"] == 1.20
    assert before.loc[(pd.Timestamp("2026-03-02", tz="UTC"), "AAA"), "eps_ttm"] == 1.30


def test_appending_future_observation_cannot_change_earlier_materialized_rows():
    baseline = _store().materialize(_index())
    future = PointInTimeFeatureObservation(
        feature_name="eps_ttm",
        value=1.80,
        observation_time="2026-03-31",
        available_time="2026-04-20",
        source="filings",
        kind=AuxiliaryFeatureKind.FUNDAMENTAL,
        symbol="AAA",
        revision_id="q1",
    )
    extended = _store((future,)).materialize(_index())
    pd.testing.assert_frame_equal(baseline, extended)


def test_coverage_gate_requires_point_in_time_and_revision_awareness():
    store = _store()
    report = store.coverage_for_index(
        _index(),
        feature_names=("policy_rate",),
        min_coverage=1.0,
    )
    assert report.research_grade_auxiliary
    assert report.coverage == 1.0

    unsafe = PointInTimeFeatureStore(
        store.observations,
        PointInTimeFeatureManifest(
            source="synthetic",
            point_in_time=False,
            revision_aware=False,
        ),
    )
    unsafe_report = unsafe.coverage_for_index(
        _index(),
        feature_names=("policy_rate",),
        min_coverage=1.0,
    )
    assert not unsafe_report.research_grade_auxiliary
    assert "auxiliary_data_not_point_in_time" in unsafe_report.reasons
    assert "auxiliary_data_not_revision_aware" in unsafe_report.reasons


def test_available_time_before_observation_time_is_rejected():
    bad = PointInTimeFeatureObservation(
        feature_name="news_sentiment",
        value=0.5,
        observation_time="2026-03-02T12:00:00Z",
        available_time="2026-03-02T11:59:00Z",
        source="news",
        kind=AuxiliaryFeatureKind.NEWS,
        symbol="AAA",
    )
    try:
        PointInTimeFeatureStore(
            (bad,),
            PointInTimeFeatureManifest(source="synthetic", point_in_time=True, revision_aware=True),
        )
    except ValueError as exc:
        assert "available_time" in str(exc)
    else:
        raise AssertionError("expected look-ahead observation to be rejected")
