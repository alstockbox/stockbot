import pandas as pd
import pytest

from stockbot.data.providers.http import ProviderError
from stockbot.data.providers.riksbank_macro import (
    BASE_URL,
    RiksbankMonetaryPolicyProvider,
    RiksbankSeries,
)


class _FakeTransport:
    def __init__(self, payload=None):
        self.payload = payload
        self.calls = []

    def get_json(self, url, headers=None):
        self.calls.append((url, headers or {}))
        return self.payload


class _LatestTransport:
    def __init__(self):
        self.calls = []

    def get_json(self, url, headers=None):
        self.calls.append((url, headers or {}))
        if url == f"{BASE_URL}/policy_rounds":
            return {"data": ["2025:4", "2026:1", "2026:2"]}
        return {"data": []}


def _payload():
    return {
        "cutoffDate": "2025-02-12T08:00:00+01:00",
        "policyRoundName": "2025:1",
        "data": [
            {"date": "2024-10-01", "value": 2.75},
            {"date": "2025-01-01", "value": 2.50},
            # Future target relative to the vintage cutoff: V1 must not collapse
            # this forecast into the realised policy_rate feature.
            {"date": "2025-04-01", "value": 2.25},
        ],
    }


def _live_vintage(round_name="2026:1", publication_date="2026-03-19"):
    return {
        "metadata": {
            "forecast_cutoff_date": "2026-02-28",
            "policy_round": round_name,
            "policy_round_end_dtm": publication_date,
        },
        "observations": [
            {"dt": "2026-01-31", "value": 2.0},
            {"dt": "2026-02-28", "value": 1.7},
            {"dt": "2026-06-30", "value": 1.5},
        ],
    }


def _live_payload():
    """Representative explicit-round production response observed 2026-09-10."""

    return {
        "data": [
            {
                "external_id": "SEMCPIFNAYNA",
                "metadata": {
                    "decimals": 2,
                    "description": "CPIF",
                    "unit": "Annual percentage change",
                },
                "vintages": _live_vintage(),
            }
        ]
    }


def _live_default_payload():
    """Representative default production response where vintages is a list."""

    latest = _live_vintage("2026:2", "2026-06-17")
    latest["metadata"]["forecast_cutoff_date"] = "2026-05-31"
    return {
        "data": [
            {
                "external_id": "SEMCPIFNAYNA",
                "metadata": {
                    "decimals": 2,
                    "description": "CPIF",
                    "unit": "Annual percentage change",
                },
                "vintages": [latest],
            }
        ]
    }


def test_fetch_series_payload_uses_documented_forecast_endpoint_and_round_query():
    transport = _FakeTransport({"data": []})
    provider = RiksbankMonetaryPolicyProvider(transport=transport)
    provider.fetch_series_payload("SEQRATENAYNA", policy_round="2025:1")

    assert len(transport.calls) == 1
    url, headers = transport.calls[0]
    assert url.startswith(BASE_URL + "?")
    assert "series=SEQRATENAYNA" in url
    assert "policy_round_name=2025%3A1" in url
    assert headers["Accept"] == "application/json"


def test_latest_policy_round_alias_resolves_to_catalogue_id_before_fetch():
    transport = _LatestTransport()
    provider = RiksbankMonetaryPolicyProvider(transport=transport)
    provider.fetch_series_payload("SEMCPIFNAYNA", policy_round="latest")

    assert len(transport.calls) == 2
    assert transport.calls[0][0] == f"{BASE_URL}/policy_rounds"
    url, _ = transport.calls[1]
    assert "series=SEMCPIFNAYNA" in url
    assert "policy_round_name=2026%3A2" in url
    assert "latest" not in url


def test_realised_rows_use_vintage_cutoff_as_available_time_and_skip_future_targets():
    provider = RiksbankMonetaryPolicyProvider(transport=_FakeTransport())
    observations = provider.observations_from_payload(
        _payload(),
        feature_name="policy_rate",
        series_id="SEQRATENAYNA",
    )

    assert len(observations) == 2
    assert [row.value for row in observations] == [2.75, 2.50]
    assert all(row.feature_name == "policy_rate" for row in observations)
    assert all(row.symbol is None for row in observations)
    assert all(row.available_time == "2025-02-12T07:00:00+00:00" for row in observations)
    assert all("series=SEQRATENAYNA" in (row.revision_id or "") for row in observations)
    assert all("round=2025:1" in (row.revision_id or "") for row in observations)


def test_live_nested_schema_uses_policy_round_publication_time_without_forecast_leakage():
    provider = RiksbankMonetaryPolicyProvider(transport=_FakeTransport())
    observations = provider.observations_from_payload(
        _live_payload(),
        feature_name="cpif_yoy",
        series_id="SEMCPIFNAYNA",
    )

    assert [row.value for row in observations] == [2.0, 1.7]
    assert all(row.available_time == "2026-03-19T00:00:00+00:00" for row in observations)
    assert all("round=2026:1" in (row.revision_id or "") for row in observations)
    assert all(row.observation_time < row.available_time for row in observations)


def test_live_default_schema_accepts_vintage_list_and_uses_latest_publication_time():
    provider = RiksbankMonetaryPolicyProvider(transport=_FakeTransport())
    observations = provider.observations_from_payload(
        _live_default_payload(),
        feature_name="cpif_yoy",
        series_id="SEMCPIFNAYNA",
    )

    assert [row.value for row in observations] == [2.0, 1.7]
    assert all(row.available_time == "2026-06-17T00:00:00+00:00" for row in observations)
    assert all("round=2026:2" in (row.revision_id or "") for row in observations)


def test_missing_cutoff_or_publication_time_fails_closed():
    provider = RiksbankMonetaryPolicyProvider(transport=_FakeTransport())
    payload = {"data": [{"date": "2025-01-01", "value": 2.5}]}

    with pytest.raises(ProviderError, match="cutoff/publication time"):
        provider.observations_from_payload(
            payload,
            feature_name="policy_rate",
            series_id="SEQRATENAYNA",
        )


def test_future_forecast_requires_explicit_forecast_parser_when_not_filtered():
    provider = RiksbankMonetaryPolicyProvider(transport=_FakeTransport())

    with pytest.raises(ProviderError, match="forecast_observations_from_payload"):
        provider.observations_from_payload(
            _payload(),
            feature_name="policy_rate",
            series_id="SEQRATENAYNA",
            realised_only=False,
        )


def test_build_store_is_point_in_time_revision_aware_and_materializes_only_after_cutoff():
    transport = _FakeTransport(_payload())
    provider = RiksbankMonetaryPolicyProvider(transport=transport)
    store = provider.build_store((RiksbankSeries("policy_rate", "SEQRATENAYNA"),), policy_round="2025:1")

    assert store.manifest.point_in_time is True
    assert store.manifest.revision_aware is True
    assert store.feature_names == ("policy_rate",)

    before = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-02-11", tz="UTC"), "AAA")],
        names=["timestamp", "symbol"],
    )
    after = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-02-13", tz="UTC"), "AAA")],
        names=["timestamp", "symbol"],
    )
    assert pd.isna(store.materialize(before).iloc[0, 0])
    assert store.materialize(after).iloc[0, 0] == pytest.approx(2.50)
