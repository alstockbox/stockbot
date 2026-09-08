import numpy as np
import pandas as pd

from stockbot.research.drift import evaluate_return_drift
from stockbot.research.ensemble import build_horizon_ensemble
from stockbot.research.multiple_testing import benjamini_hochberg, evaluate_discoveries
from stockbot.research.regime_eval import build_market_regime_series, evaluate_regime_performance


def _bars() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=180, freq="B", tz="UTC")
    rows = []
    for j, symbol in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        first = np.full(60, 0.003 + j * 0.0001)
        second = np.full(60, -0.003 - j * 0.0001)
        third = np.full(60, 0.0001)
        returns = np.concatenate([first, second, third])
        close = 100.0 * np.cumprod(1.0 + returns)
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": close[i],
                    "high": close[i] * 1.01,
                    "low": close[i] * 0.99,
                    "close": close[i],
                    "volume": 1_000_000 + i * 100,
                }
            )
    return pd.DataFrame(rows)


def test_regime_series_is_causal_and_regime_report_scores_available_segments():
    bars = _bars()
    full = build_market_regime_series(bars)
    cutoff = full.index[139]
    short = build_market_regime_series(bars.loc[pd.to_datetime(bars["timestamp"], utc=True) <= cutoff])
    pd.testing.assert_series_equal(full.loc[short.index], short)

    returns = pd.Series(0.001, index=full.index)
    report = evaluate_regime_performance(returns, full, min_observations=5)
    assert 0.0 <= report.score <= 1.0
    assert 0.0 <= report.coverage <= 1.0
    assert sum(report.observations_by_regime.values()) == len(full)


def test_bh_adjustment_is_monotone_and_discovery_control_penalizes_noise():
    q = benjamini_hochberg([0.001, 0.02, 0.25, 0.8])
    assert q[0] <= q[1] <= q[2] <= q[3]

    rng = np.random.default_rng(7)
    index = pd.date_range("2025-01-01", periods=252, freq="B")
    good = pd.Series(rng.normal(0.0025, 0.006, len(index)), index=index)
    noise = pd.Series(rng.normal(0.0, 0.01, len(index)), index=index)
    discoveries = evaluate_discoveries({"good": good, "noise": noise}, max_q_value=0.20)
    assert discoveries["good"].q_value <= discoveries["noise"].q_value
    assert discoveries["good"].confidence >= discoveries["noise"].confidence


def test_horizon_ensemble_blends_members_and_produces_stress_metrics():
    index = pd.date_range("2025-01-01", periods=180, freq="B", tz="UTC")
    r1 = pd.Series(0.0008, index=index)
    r2 = pd.Series(0.0005, index=index)
    regime = pd.Series("neutral_chop", index=index)
    report = build_horizon_ensemble(
        {"h1": r1, "h20": r2},
        {"h1": 1.5, "h20": 1.0},
        regime_series=regime,
    )
    assert set(report.member_weights) == {"h1", "h20"}
    assert abs(sum(report.member_weights.values()) - 1.0) < 1e-9
    assert len(report.returns) == len(index)
    assert 0.0 <= report.stress_report.score <= 1.0
    assert 0.0 <= report.score <= 1.0


def test_drift_detector_flags_recent_degradation():
    rng = np.random.default_rng(9)
    index = pd.date_range("2024-01-01", periods=240, freq="B")
    stable = rng.normal(0.0012, 0.006, 170)
    degraded = rng.normal(-0.0025, 0.018, 70)
    returns = pd.Series(np.concatenate([stable, degraded]), index=index)
    report = evaluate_return_drift(returns)
    assert report.degraded
    assert report.score < 1.0
    assert report.sharpe_delta < 0.0
