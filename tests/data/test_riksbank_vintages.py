from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from stockbot.data.providers.http import ProviderError
from stockbot.data.providers.riksbank_macro import (
    BASE_URL,
    RiksbankMonetaryPolicyProvider,
    RiksbankSeries,
    policy_round_names,
)


class _VintageTransport:
    def __init__(self):
        self.calls = []

    def get_json(self, url, headers=None):
        self.calls.append(url)
        if url == f"{BASE_URL}/policy_rounds":
            return {
                "policyRounds": [
                    {"policyRoundName": "2021:2"},
                    {"policyRoundName": "2020:4"},
                    {"policyRoundName": "2021:1"},
                ]
            }

        query = parse_qs(urlparse(url).query)
        assert query == {"series": ["SEQRATENAYNA"]}, (
            "historical vintage collection must fetch the series once without "
            "a policy-round query filter"
        )
        return {
            "data": [
                {
                    "external_id": "SEQRATENAYNA",
                    "vintages": [
                        {
                            "metadata": {
                                "forecast_cutoff_date": "2020-11-26",
                                "policy_round": "2020:4",
                                "policy_round_end_dtm": "2020-11-26",
                            },
                            "observations": [{"dt": "2020-10-01", "value": 0.10}],
                        },
                        {
                            "metadata": {
                                "forecast_cutoff_date": "2021-02-10",
                                "policy_round": "2021:1",
                                "policy_round_end_dtm": "2021-02-10",
                            },
                            "observations": [{"dt": "2020-10-01", "value": 0.25}],
                        },
                        {
                            "metadata": {
                                "forecast_cutoff_date": "2021-04-26",
                                "policy_round": "2021:2",
                                "policy_round_end_dtm": "2021-04-26",
                            },
                            "observations": [{"dt": "2020-10-01", "value": 0.50}],
                        },
                    ],
                }
            ]
        }


def test_policy_round_parser_is_strict_deduplicated_and_chronological():
    payload = {
        "policyRounds": [
            {"policyRoundName": "2025:3"},
            {"policyRoundName": "2024:4"},
            {"policyRoundName": "2025:1"},
            {"policyRoundName": "2025:1"},
        ]
    }
    assert policy_round_names(payload) == ("2024:4", "2025:1", "2025:3")

    with pytest.raises(ProviderError, match="invalid policy-round identifier"):
        policy_round_names(["2025-Q1"])


def test_policy_round_year_filter_uses_historical_endpoint():
    provider = RiksbankMonetaryPolicyProvider(transport=_VintageTransport())
    assert provider.policy_round_names(start_year=2021, end_year=2021) == ("2021:1", "2021:2")


def test_vintage_store_replays_revision_only_after_later_cutoff():
    transport = _VintageTransport()
    provider = RiksbankMonetaryPolicyProvider(transport=transport)
    store = provider.build_vintage_store(
        (RiksbankSeries("policy_rate", "SEQRATENAYNA"),),
        start_year=2021,
        end_year=2021,
    )

    first_vintage = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-03-01", tz="UTC"), "AAA")],
        names=["timestamp", "symbol"],
    )
    second_vintage = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-05-03", tz="UTC"), "AAA")],
        names=["timestamp", "symbol"],
    )

    assert store.materialize(first_vintage).iloc[0, 0] == pytest.approx(0.25)
    assert store.materialize(second_vintage).iloc[0, 0] == pytest.approx(0.50)
    assert store.manifest.point_in_time is True
    assert store.manifest.revision_aware is True

    series_calls = [url for url in transport.calls if "series=SEQRATENAYNA" in url]
    assert len(series_calls) == 1
    assert "policy_round_name" not in series_calls[0]
