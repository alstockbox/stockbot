import numpy as np
import pandas as pd

from stockbot.research.factor_exposure import build_internal_factor_returns, evaluate_factor_exposure


def _bars() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.date_range("2024-01-02", periods=180, freq="B", tz="UTC")
    rows = []
    market = rng.normal(0.0003, 0.005, len(dates))
    for j, symbol in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")):
        local = market + 0.0004 * np.sin(np.arange(len(dates)) / (8 + j)) + rng.normal(0.0, 0.003, len(dates))
        close = 100.0 * np.exp(np.cumsum(local))
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": float(close[i]),
                    "high": float(close[i] * 1.005),
                    "low": float(close[i] * 0.995),
                    "close": float(close[i]),
                    "volume": 500_000 + 100_000 * j,
                }
            )
    return pd.DataFrame(rows)


def test_internal_factor_returns_are_finite_and_aligned():
    factors = build_internal_factor_returns(_bars())
    assert list(factors.columns) == ["market", "momentum", "liquidity", "volatility"]
    assert len(factors) == 180
    assert np.isfinite(factors.to_numpy()).all()


def test_factor_exposure_identifies_market_beta_and_residual_metrics():
    factors = build_internal_factor_returns(_bars())
    rng = np.random.default_rng(19)
    strategy = 0.8 * factors["market"] + 0.3 * factors["momentum"] + pd.Series(
        rng.normal(0.0002, 0.0015, len(factors)), index=factors.index
    )
    report = evaluate_factor_exposure(strategy, factors)

    assert report.observations == len(factors)
    assert 0.0 <= report.r_squared <= 1.0
    assert 0.0 <= report.idiosyncratic_score <= 1.0
    assert abs(report.factor_betas["market"] - 0.8) < 0.25
    assert abs(report.factor_betas["momentum"] - 0.3) < 0.25
    assert np.isfinite(report.residual_metrics["sharpe"])
