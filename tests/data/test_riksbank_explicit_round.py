from urllib.parse import parse_qs, urlparse

from stockbot.data.providers.riksbank_macro import (
    RiksbankMonetaryPolicyProvider,
    RiksbankSeries,
)


class _ExplicitRoundTransport:
    def __init__(self):
        self.calls = []

    def get_json(self, url, headers=None):
        self.calls.append(url)
        query = parse_qs(urlparse(url).query)
        assert query == {"series": ["SEQRATENAYNA"]}, (
            "explicit policy-round selection must fetch the bulk series payload "
            "without policy_round_name"
        )
        return {
            "data": [
                {
                    "external_id": "SEQRATENAYNA",
                    "vintages": [
                        {
                            "metadata": {
                                "forecast_cutoff_date": "2025-02-12",
                                "policy_round": "2025:1",
                                "policy_round_end_dtm": "2025-02-12",
                            },
                            "observations": [{"dt": "2024-10-01", "value": 2.50}],
                        },
                        {
                            "metadata": {
                                "forecast_cutoff_date": "2025-05-08",
                                "policy_round": "2025:2",
                                "policy_round_end_dtm": "2025-05-08",
                            },
                            "observations": [{"dt": "2024-10-01", "value": 2.25}],
                        },
                    ],
                }
            ]
        }


def test_build_store_selects_explicit_policy_round_locally():
    transport = _ExplicitRoundTransport()
    provider = RiksbankMonetaryPolicyProvider(transport=transport)

    store = provider.build_store(
        (RiksbankSeries("policy_rate", "SEQRATENAYNA"),),
        policy_round="2025:2",
    )

    assert len(transport.calls) == 1
    assert "policy_round_name" not in transport.calls[0]
    assert len(store.observations) == 1
    observation = store.observations[0]
    assert observation.value == 2.25
    assert observation.available_time == "2025-05-08T00:00:00+00:00"
    assert observation.revision_id == "series=SEQRATENAYNA|round=2025:2"
