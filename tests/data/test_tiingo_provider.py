import pytest

from stockbot.data.providers.tiingo import ProviderError, TiingoProvider


class FakeTransport:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
    def get_json(self, url, headers):
        self.calls.append((url, headers))
        return self.payload


def _payload():
    return [{
        "date": "2026-01-02T00:00:00.000Z",
        "open": 100.0, "high": 103.0, "low": 99.0, "close": 102.0, "volume": 1000,
        "adjOpen": 50.0, "adjHigh": 51.5, "adjLow": 49.5, "adjClose": 51.0, "adjVolume": 2000,
        "divCash": 0.5, "splitFactor": 2.0,
    }]


def test_tiingo_requires_token_before_request():
    with pytest.raises(ProviderError, match="token"):
        TiingoProvider(token="")


def test_tiingo_builds_authorized_request_and_normalizes_all_price_fields():
    transport = FakeTransport(_payload())
    provider = TiingoProvider(token="secret-token", transport=transport)
    bars = provider.fetch_bars("AAPL", "2026-01-01", "2026-01-31")
    url, headers = transport.calls[0]
    assert "startDate=2026-01-01" in url and "endDate=2026-01-31" in url
    assert headers["Authorization"] == "Token secret-token"
    row = bars.iloc[0]
    assert row["symbol"] == "AAPL"
    assert row["close"] == 102.0 and row["adj_close"] == 51.0
    assert row["div_cash"] == 0.5 and row["split_factor"] == 2.0
    assert row["provider"] == "tiingo"


def test_tiingo_rejects_empty_or_malformed_payload():
    for payload in ([], {"bad": "shape"}, [{"date": "2026-01-02"}]):
        provider = TiingoProvider(token="x", transport=FakeTransport(payload))
        with pytest.raises(ProviderError):
            provider.fetch_bars("AAPL", "2026-01-01", "2026-01-31")
