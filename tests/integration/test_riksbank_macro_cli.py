from stockbot.cli.riksbank_macro import _parse_series, build_parser, run_from_args
from stockbot.data.point_in_time_features import (
    AuxiliaryFeatureKind,
    PointInTimeFeatureManifest,
    PointInTimeFeatureObservation,
    PointInTimeFeatureStore,
)
from stockbot.data.research_inputs import load_point_in_time_feature_store


class _FakeProvider:
    def __init__(self):
        self.calls = []

    @staticmethod
    def _store():
        return PointInTimeFeatureStore(
            (
                PointInTimeFeatureObservation(
                    feature_name="policy_rate",
                    value=2.5,
                    observation_time="2025-01-01T00:00:00Z",
                    available_time="2025-02-12T07:00:00Z",
                    source="riksbank-monetary-policy",
                    kind=AuxiliaryFeatureKind.MACRO,
                    revision_id="series=SEQRATENAYNA|round=2025:1",
                ),
            ),
            PointInTimeFeatureManifest(
                source="riksbank-monetary-policy",
                point_in_time=True,
                revision_aware=True,
                available_time_semantics="riksbank_cutoff_or_publication_time",
            ),
        )

    def build_vintage_store(self, series, *, start_year, end_year, include_forecasts=False):
        self.calls.append(
            (
                "vintages",
                tuple(item.feature_name for item in series),
                start_year,
                end_year,
                bool(include_forecasts),
            )
        )
        return self._store()

    def build_store(self, series, *, policy_round, include_forecasts=False):
        self.calls.append(
            (
                "round",
                tuple(item.feature_name for item in series),
                policy_round,
                bool(include_forecasts),
            )
        )
        return self._store()


def test_riksbank_cli_writes_fingerprinted_research_input(tmp_path):
    parser = build_parser()
    output = tmp_path / "riksbank.json"
    args = parser.parse_args(
        [
            "--output",
            str(output),
            "--series",
            "policy_rate",
            "--start-year",
            "2021",
            "--end-year",
            "2025",
        ]
    )
    provider = _FakeProvider()

    assert run_from_args(args, provider=provider) == 0
    assert provider.calls == [("vintages", ("policy_rate",), 2021, 2025, False)]

    loaded = load_point_in_time_feature_store(output)
    assert loaded.feature_names == ("policy_rate",)
    assert loaded.manifest.point_in_time is True
    assert loaded.manifest.revision_aware is True
    assert loaded.fingerprint == _FakeProvider._store().fingerprint


def test_riksbank_cli_single_round_uses_round_mode(tmp_path):
    parser = build_parser()
    output = tmp_path / "round.json"
    args = parser.parse_args(
        [
            "--output",
            str(output),
            "--series",
            "policy_rate",
            "--policy-round",
            "2025:1",
        ]
    )
    provider = _FakeProvider()

    assert run_from_args(args, provider=provider) == 0
    assert provider.calls == [("round", ("policy_rate",), "2025:1", False)]


def test_riksbank_cli_include_forecasts_is_explicit_opt_in(tmp_path):
    parser = build_parser()
    output = tmp_path / "forecast.json"
    args = parser.parse_args(
        [
            "--output",
            str(output),
            "--series",
            "policy_rate,cpif_yoy",
            "--start-year",
            "2022",
            "--include-forecasts",
        ]
    )
    provider = _FakeProvider()

    assert args.include_forecasts is True
    assert run_from_args(args, provider=provider) == 0
    assert provider.calls == [
        ("vintages", ("policy_rate", "cpif_yoy"), 2022, None, True)
    ]


def test_riksbank_series_alias_parser_rejects_unknown_and_duplicates():
    assert tuple(item.feature_name for item in _parse_series("policy_rate,cpif_yoy")) == (
        "policy_rate",
        "cpif_yoy",
    )

    try:
        _parse_series("policy_rate,policy_rate")
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate aliases must be rejected")

    try:
        _parse_series("not_a_series")
    except ValueError as exc:
        assert "unknown Riksbank feature aliases" in str(exc)
    else:
        raise AssertionError("unknown aliases must be rejected")
