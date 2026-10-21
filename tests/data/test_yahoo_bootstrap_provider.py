import pandas as pd
import pytest

from stockbot.data.providers.yahoo_bootstrap import ProviderError, YahooBootstrapProvider
from stockbot.data.schemas import DataGrade


class FakeTransport:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
    def get_json(self, url, headers):
        self.calls.append((url, headers))
        return self.payload


def _payload():
    ts = int(pd.Timestamp("2026-01-02", tz="UTC").timestamp())
    return {"chart": {"result": [{
        "timestamp": [ts],
        "indicators": {
            "quote": [{"open": [100.0], "high": [103.0], "low": [99.0], "close": [102.0], "volume": [1000]}],
            "adjclose": [{"adjclose": [101.5]}],
        },
        "events": {
            "dividends": {str(ts): {"amount": 0.4}},
            "splits": {str(ts): {"numerator": 2.0, "denominator": 1.0}},
        },
    }], "error": None}}


def test_yahoo_bootstrap_is_permanently_bootstrap_grade_and_parses_chart():
    transport = FakeTransport(_payload())
    provider = YahooBootstrapProvider(transport=transport)
    bars = provider.fetch_bars("AAPL", "2026-01-01", "2026-01-31")
    assert provider.grade is DataGrade.BOOTSTRAP
    url, _ = transport.calls[0]
    assert "period1=" in url and "period2=" in url and "interval=1d" in url
    row = bars.iloc[0]
    assert row["close"] == 102.0
    assert row["adj_close"] == 101.5
    assert row["div_cash"] == 0.4
    assert row["split_factor"] == 2.0
    assert row["provider"] == "yahoo-bootstrap"


def test_yahoo_bootstrap_rejects_missing_chart_result():
    provider = YahooBootstrapProvider(transport=FakeTransport({"chart": {"result": None, "error": None}}))
    with pytest.raises(ProviderError):
        provider.fetch_bars("AAPL", "2026-01-01", "2026-01-31")
