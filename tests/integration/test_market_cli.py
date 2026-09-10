import os
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from stockbot.cli.market_training import (
    _evaluate_factory_data_quality,
    build_parser,
    resolve_provider,
)
from stockbot.data.providers.http import ProviderError
from stockbot.data.providers.yahoo_bootstrap import YahooBootstrapProvider
from stockbot.data.schemas import DataGrade
from stockbot.data.universe import PointInTimeUniverse, UniverseManifest, UniverseMembership


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


def test_cli_parser_accepts_research_factory_specialist_options(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "--provider", "yahoo-bootstrap",
        "--symbols", "AAPL,MSFT,NVDA",
        "--start", "2020-01-01",
        "--end", "2025-12-31",
        "--snapshot-root", str(tmp_path),
        "--factory",
        "--factory-regime-specialists",
        "--factory-specialists-top-k", "3",
        "--factory-run-dir", str(tmp_path / "runs"),
    ])
    assert args.factory is True
    assert args.factory_regime_specialists is True
    assert args.factory_specialists_top_k == 3
    assert args.factory_run_dir == str(tmp_path / "runs")


def test_cli_parser_accepts_point_in_time_research_inputs(tmp_path):
    parser = build_parser()
    feature_path = tmp_path / "features.json"
    universe_path = tmp_path / "universe.json"
    args = parser.parse_args([
        "--provider", "yahoo-bootstrap",
        "--symbols", "AAPL,MSFT,NVDA",
        "--start", "2020-01-01",
        "--end", "2025-12-31",
        "--snapshot-root", str(tmp_path),
        "--factory",
        "--factory-auxiliary-input", str(feature_path),
        "--factory-auxiliary-features", "eps_ttm,policy_rate",
        "--factory-auxiliary-max-age-days", "120",
        "--factory-auxiliary-min-coverage", "0.9",
        "--factory-universe-input", str(universe_path),
        "--factory-deep-diagnostics",
    ])
    assert args.factory_auxiliary_input == str(feature_path)
    assert args.factory_auxiliary_features == "eps_ttm,policy_rate"
    assert args.factory_auxiliary_max_age_days == 120
    assert args.factory_auxiliary_min_coverage == pytest.approx(0.9)
    assert args.factory_universe_input == str(universe_path)
    assert args.factory_deep_diagnostics is True


def test_cli_parser_accepts_explicit_research_data_attestations():
    parser = build_parser()
    args = parser.parse_args([
        "--provider", "yahoo-bootstrap",
        "--symbols", "AAPL,MSFT",
        "--start", "2020-01-01",
        "--end", "2025-12-31",
        "--factory",
        "--factory-attest-adjusted-prices-verified",
        "--factory-attest-corporate-actions-complete",
        "--factory-attest-corporate-actions-point-in-time",
    ])
    assert args.factory_attest_adjusted_prices_verified is True
    assert args.factory_attest_corporate_actions_complete is True
    assert args.factory_attest_corporate_actions_point_in_time is True


def _quality_bars() -> pd.DataFrame:
    rows = []
    for symbol in ("AAA", "BBB"):
        for dt in pd.date_range("2025-01-02", periods=3, freq="B", tz="UTC"):
            rows.append(
                {
                    "timestamp": dt,
                    "symbol": symbol,
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.0,
                    "close": 101.0,
                    "volume": 1_000_000.0,
                    "adj_open": 100.0,
                    "adj_high": 102.0,
                    "adj_low": 99.0,
                    "adj_close": 101.0,
                    "adj_volume": 1_000_000.0,
                    "div_cash": 0.0,
                    "split_factor": 1.0,
                    "provider": "verified-test",
                    "retrieved_at": pd.Timestamp("2025-01-10T20:00:00Z"),
                }
            )
    return pd.DataFrame(rows)


def _quality_universe() -> PointInTimeUniverse:
    return PointInTimeUniverse(
        (
            UniverseMembership("AAA", "2020-01-01"),
            UniverseMembership("BBB", "2020-01-01"),
            UniverseMembership("OLD", "2020-01-01", "2024-12-31", delisted=True),
        ),
        UniverseManifest(
            source="verified-universe",
            survivorship_bias_controlled=True,
            includes_delisted_securities=True,
            point_in_time_membership=True,
        ),
    )


def _quality_args() -> SimpleNamespace:
    return SimpleNamespace(
        factory_attest_adjusted_prices_verified=True,
        factory_attest_corporate_actions_complete=True,
        factory_attest_corporate_actions_point_in_time=True,
    )


def test_factory_quality_helper_fails_closed_without_point_in_time_universe():
    snapshot = SimpleNamespace(
        bars=_quality_bars(),
        manifest=SimpleNamespace(grade=DataGrade.RESEARCH_GRADE),
    )
    report, fingerprint, effective_grade = _evaluate_factory_data_quality(
        snapshot,
        None,
        _quality_args(),
    )
    assert not report.research_grade_eligible
    assert "point_in_time_universe_missing" in report.reasons
    assert len(fingerprint) == 64
    assert effective_grade is DataGrade.BOOTSTRAP


def test_factory_quality_helper_never_upgrades_bootstrap_declared_grade():
    snapshot = SimpleNamespace(
        bars=_quality_bars(),
        manifest=SimpleNamespace(grade=DataGrade.BOOTSTRAP),
    )
    report, fingerprint, effective_grade = _evaluate_factory_data_quality(
        snapshot,
        _quality_universe(),
        _quality_args(),
    )
    assert report.research_grade_eligible
    assert len(fingerprint) == 64
    assert effective_grade is DataGrade.BOOTSTRAP


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
