import pandas as pd

from stockbot.data.universe import PointInTimeUniverse, UniverseManifest, UniverseMembership


def _universe(controlled: bool = True) -> PointInTimeUniverse:
    return PointInTimeUniverse(
        [
            UniverseMembership("AAA", "2024-01-01", exchange="XSTO", sector="Industrials"),
            UniverseMembership("BBB", "2024-03-01", exchange="XSTO", sector="Technology"),
            UniverseMembership(
                "OLD",
                "2024-01-01",
                "2024-07-01",
                exchange="XSTO",
                sector="Technology",
                delisted=True,
            ),
        ],
        UniverseManifest(
            source="synthetic-point-in-time-test",
            survivorship_bias_controlled=controlled,
            includes_delisted_securities=controlled,
            point_in_time_membership=controlled,
        ),
    )


def test_members_at_respects_historical_membership_and_delisting():
    universe = _universe()
    assert universe.members_at("2024-02-01") == ("AAA", "OLD")
    assert universe.members_at("2024-04-01") == ("AAA", "BBB", "OLD")
    assert universe.members_at("2024-08-01") == ("AAA", "BBB")
    assert universe.metadata_at("OLD", "2024-06-01").delisted
    assert universe.metadata_at("OLD", "2024-08-01") is None


def test_research_grade_universe_requires_point_in_time_survivorship_controls():
    bars = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": "2024-02-01"},
            {"symbol": "OLD", "timestamp": "2024-02-01"},
            {"symbol": "BBB", "timestamp": "2024-04-01"},
            {"symbol": "OLD", "timestamp": "2024-06-01"},
        ]
    )
    good = _universe(True).coverage_for_bars(bars)
    bad = _universe(False).coverage_for_bars(bars)

    assert good.membership_coverage == 1.0
    assert good.research_grade_universe
    assert good.reasons == ()
    assert not bad.research_grade_universe
    assert "survivorship_bias_not_controlled" in bad.reasons
    assert "delisted_securities_not_included" in bad.reasons


def test_missing_historical_membership_is_reported():
    bars = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": "2024-02-01"},
            {"symbol": "UNKNOWN", "timestamp": "2024-02-01"},
        ]
    )
    report = _universe().coverage_for_bars(bars, min_membership_coverage=0.99)
    assert report.membership_coverage == 0.5
    assert report.missing_symbols == ("UNKNOWN",)
    assert not report.research_grade_universe
