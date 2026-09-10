import os
import subprocess
import sys

import pytest

from stockbot.cli.market_training import build_parser, resolve_provider
from stockbot.data.providers.http import ProviderError
from stockbot.data.providers.yahoo_bootstrap import YahooBootstrapProvider


def test_cli_parser_accepts_required_market_download_arguments(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "--provider", "yahoo-bootstrap",
        "--symbols", "AAPL,MSFT",
        "--start", "2020-01-01",
        "--end", "2025-12-31",
        "--snapshot-root", str(tmp_path),
        "--train",
    ])
    assert args.provider == "yahoo-bootstrap"
    assert args.symbols == "AAPL,MSFT"
    assert args.train is True


def test_resolve_provider_requires_tiingo_token(monkeypatch):
    monkeypatch.delenv("TIINGO_API_TOKEN", raising=False)
    with pytest.raises(ProviderError, match="TIINGO_API_TOKEN"):
        resolve_provider("tiingo")


def test_resolve_provider_creates_zero_key_bootstrap_provider():
    assert isinstance(resolve_provider("yahoo-bootstrap"), YahooBootstrapProvider)


def test_script_fails_before_network_when_tiingo_token_missing(tmp_path):
    env = os.environ.copy()
    env.pop("TIINGO_API_TOKEN", None)
    result = subprocess.run(
        [sys.executable, "scripts/run_market_training.py", "--provider", "tiingo", "--symbols", "AAPL", "--start", "2026-01-01", "--end", "2026-01-31", "--snapshot-root", str(tmp_path)],
        cwd=os.getcwd(), env=env, text=True, capture_output=True,
    )
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "tiingo_api_token" in combined
    assert "authorization" not in combined
