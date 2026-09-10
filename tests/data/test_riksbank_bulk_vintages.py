from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from stockbot.data.providers.riksbank_macro import (
    BASE_URL,
    RiksbankMonetaryPolicyProvider,
    RiksbankSeries,
)


class _BulkVintageTransport:
    def __init__(self):
        self.calls = []

    def get_json(self, url, headers=None):
        self.calls.append(url)
        if url == f"{BASE_URL}/policy_rounds":
            return {
                "policyRounds": [
                    {"policyRoundName": "2021:1"},
                    {"policyRoundName": "2021:2"},
                    {"policyRoundName": "2022:1"},
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
                        {
                            "metadata": {
                                "forecast_cutoff_date": "2022-02-10",
                                "policy_round": "2022:1",
                                "policy_round_end_dtm": "2022-02-10",
                            },
                            "observations": [{"dt": "2020-10-01", "value": 0.75}],
                        },
                    ],
                }
            ]
        }


def test_vintage_store_fetches_each_series_once_and_filters_rounds_locally():
    transport = _BulkVintageTransport()
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
    after_excluded_vintage = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2022-03-01", tz="UTC"), "AAA")],
        names=["timestamp", "symbol"],
    )

    assert store.materialize(first_vintage).iloc[0, 0] == pytest.approx(0.25)
    assert store.materialize(second_vintage).iloc[0, 0] == pytest.approx(0.50)
    assert store.materialize(after_excluded_vintage).iloc[0, 0] == pytest.approx(0.50)

    series_calls = [url for url in transport.calls if "series=SEQRATENAYNA" in url]
    assert len(series_calls) == 1
    assert "policy_round_name" not in series_calls[0]
