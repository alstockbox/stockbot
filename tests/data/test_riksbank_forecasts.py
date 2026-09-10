import pandas as pd
import pytest

from stockbot.data.providers.riksbank_macro import (
    RiksbankForecastHorizon,
    RiksbankMonetaryPolicyProvider,
    RiksbankSeries,
)


class _PayloadTransport:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get_json(self, url, headers=None):
        self.calls.append(url)
        return self.payload


def _forecast_payload():
    return {
        "cutoffDate": "2025-02-12T08:00:00+01:00",
        "policyRoundName": "2025:1",
        "data": [
            {"date": "2025-01-01", "value": 2.50},
            {"date": "2025-05-15", "value": 2.25},
            {"date": "2025-08-15", "value": 2.00},
            {"date": "2026-02-15", "value": 1.75},
        ],
    }


def test_forecast_rows_are_mapped_to_stable_fixed_horizon_features():
    provider = RiksbankMonetaryPolicyProvider(transport=_PayloadTransport(_forecast_payload()))
    rows = provider.forecast_observations_from_payload(
        _forecast_payload(),
        feature_name="policy_rate",
        series_id="SEQRATENAYNA",
    )

    assert [row.feature_name for row in rows] == [
        "policy_rate__forecast_3m",
        "policy_rate__forecast_6m",
        "policy_rate__forecast_12m",
    ]
    assert [row.value for row in rows] == [2.25, 2.00, 1.75]
    assert all(row.observation_time == "2025-02-12T07:00:00+00:00" for row in rows)
    assert all(row.available_time == "2025-02-12T07:00:00+00:00" for row in rows)
    assert "target=2025-05-15T00:00:00+00:00" in (rows[0].revision_id or "")
    assert "horizon=3m" in (rows[0].revision_id or "")


def test_forecast_horizon_chooses_nearest_target_deterministically():
    payload = {
        "cutoffDate": "2025-01-01T00:00:00Z",
        "policyRoundName": "2025:1",
        "data": [
            {"date": "2025-03-20", "value": 1.0},
            {"date": "2025-04-05", "value": 2.0},
        ],
    }
    provider = RiksbankMonetaryPolicyProvider(transport=_PayloadTransport(payload))
    rows = provider.forecast_observations_from_payload(
        payload,
        feature_name="x",
        series_id="SERIES",
        horizons=(RiksbankForecastHorizon("3m", 91, 30),),
    )
    assert len(rows) == 1
    assert rows[0].value == pytest.approx(2.0)


def test_build_store_can_include_realised_and_forecast_features_in_one_call():
    transport = _PayloadTransport(_forecast_payload())
    provider = RiksbankMonetaryPolicyProvider(transport=transport)
    store = provider.build_store(
        (RiksbankSeries("policy_rate", "SEQRATENAYNA"),),
        policy_round="2025:1",
        include_forecasts=True,
    )

    assert transport.calls and len(transport.calls) == 1
    assert store.feature_names == (
        "policy_rate",
        "policy_rate__forecast_12m",
        "policy_rate__forecast_3m",
        "policy_rate__forecast_6m",
    )

    asof = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-02-13", tz="UTC"), "AAA")],
        names=["timestamp", "symbol"],
    )
    frame = store.materialize(asof)
    assert frame.loc[(pd.Timestamp("2025-02-13", tz="UTC"), "AAA"), "policy_rate"] == pytest.approx(2.50)
    assert frame.loc[(pd.Timestamp("2025-02-13", tz="UTC"), "AAA"), "policy_rate__forecast_3m"] == pytest.approx(2.25)


def test_forecast_horizon_validation_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="positive"):
        RiksbankForecastHorizon("bad", 0, 10)
    with pytest.raises(ValueError, match="cannot be negative"):
        RiksbankForecastHorizon("bad", 90, -1)
