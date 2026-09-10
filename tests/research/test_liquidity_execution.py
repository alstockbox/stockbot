import numpy as np
import pandas as pd

from stockbot.data.panel import build_panel
from stockbot.research.liquidity_execution import LiquidityExecutionConfig, simulate_liquidity_aware_execution


def _bars(volume: int) -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=80, freq="B", tz="UTC")
    rows = []
    for j, symbol in enumerate(["AAA", "BBB", "CCC"]):
        close = 100.0 * np.cumprod(np.full(len(dates), 1.0002 + j * 0.0001))
        for i, dt in enumerate(dates):
            rows.append({
                "symbol": symbol,
                "timestamp": dt,
                "open": close[i],
                "high": close[i] * 1.005,
                "low": close[i] * 0.995,
                "close": close[i],
                "volume": volume,
            })
    return pd.DataFrame(rows)


def _predictions(bars: pd.DataFrame) -> pd.Series:
    panel = build_panel(bars)
    predictions = pd.Series(np.nan, index=panel.index, dtype=float)
    dates = panel.index.get_level_values("timestamp")
    symbols = panel.index.get_level_values("symbol")
    active = dates >= pd.Timestamp("2025-01-20", tz="UTC")
    score_map = {"AAA": 0.1, "BBB": 0.4, "CCC": 1.0}
    predictions.loc[active] = symbols[active].map(score_map).astype(float)
    return predictions


def test_illiquid_high_capital_strategy_gets_partial_fills_and_participation_cap():
    bars = _bars(volume=100)
    config = LiquidityExecutionConfig(
        capital=1_000_000.0,
        max_participation=0.05,
        adv_window=5,
        impact_bps_at_one_pct_adv=20.0,
    )
    report = simulate_liquidity_aware_execution(
        bars,
        _predictions(bars),
        top_fraction=0.34,
        config=config,
    )

    assert report.partial_fill_fraction > 0.5
    assert report.max_participation <= config.max_participation + 1e-9
    assert report.metrics["cost_ratio"] > 0.0
    assert report.average_tracking_error > 0.0


def test_liquid_small_capital_strategy_can_fill_target_more_completely():
    bars = _bars(volume=1_000_000)
    config = LiquidityExecutionConfig(
        capital=10_000.0,
        max_participation=0.05,
        adv_window=5,
    )
    report = simulate_liquidity_aware_execution(
        bars,
        _predictions(bars),
        top_fraction=0.34,
        config=config,
    )

    assert report.partial_fill_fraction < 0.05
    assert report.average_tracking_error < 0.05
    assert 0.0 <= report.stress_report.score <= 1.0
