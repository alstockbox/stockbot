import importlib
import json

import pandas as pd
import pytest

from stockbot.data.schemas import DataGrade


def test_provider_smoke_reports_yahoo_and_skips_tiingo_without_token(tmp_path, monkeypatch):
    try:
        provider_smoke = importlib.import_module("stockbot.cli.provider_smoke")
    except ModuleNotFoundError:
        pytest.fail("provider smoke CLI is not implemented")

    class FakeYahooProvider:
        name = "yahoo-bootstrap"
        grade = DataGrade.BOOTSTRAP

        def fetch_bars(self, symbol, start, end):
            return pd.DataFrame(
                {
                    "symbol": [symbol, symbol],
                    "timestamp": pd.to_datetime(["2026-09-01", "2026-09-02"], utc=True),
                }
            )

    monkeypatch.setattr(provider_smoke, "YahooBootstrapProvider", FakeYahooProvider)
    monkeypatch.delenv("TIINGO_API_TOKEN", raising=False)

    report_path = tmp_path / "provider-smoke.json"
    rc = provider_smoke.main(
        [
            "--symbol",
            "AAPL",
            "--start",
            "2026-09-01",
            "--end",
            "2026-09-05",
            "--report",
            str(report_path),
        ]
    )

    assert rc == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["symbol"] == "AAPL"
    assert payload["broker_execution_available"] is False

    yahoo = payload["providers"]["yahoo-bootstrap"]
    assert yahoo["status"] == "ok"
    assert yahoo["grade"] == DataGrade.BOOTSTRAP.value
    assert yahoo["rows"] == 2
    assert yahoo["first_timestamp"].startswith("2026-09-01")
    assert yahoo["last_timestamp"].startswith("2026-09-02")

    tiingo = payload["providers"]["tiingo"]
    assert tiingo["status"] == "skipped"
    assert "TIINGO_API_TOKEN" in tiingo["reason"]
