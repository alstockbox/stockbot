from datetime import datetime, timedelta, timezone

from stockbot.data.live_models import MarketBar
from stockbot.research.live_signal import LiveSignalConfig, LiveSignalEngine
from stockbot.runtime.live_planner import LivePlannerConfig, LiveShadowPlanner


def trending_bars(count=140):
    start = datetime(2026, 9, 20, tzinfo=timezone.utc)
    price = 100.0
    bars = []
    for index in range(count):
        open_price = price
        price *= 1.001
        bars.append(
            MarketBar(
                symbol="TEST",
                timestamp=start + timedelta(minutes=5 * index),
                open=open_price,
                high=price * 1.001,
                low=open_price * 0.999,
                close=price,
                volume=1000.0 + index,
            )
        )
    return bars


class BarsProvider:
    def __init__(self, bars):
        self._bars = bars

    def bars(self, symbol, timeframe=None, count=250):
        return self._bars[-count:]


def test_live_signal_uses_only_historical_matched_forward_returns():
    bars = trending_bars()
    engine = LiveSignalEngine(
        LiveSignalConfig(
            min_bars=80,
            similarity_band=0.20,
            min_edge_samples=12,
        )
    )
    result = engine.analyze("TEST", bars)

    assert result.sample_count >= 12
    assert result.expected_return > 0.0
    assert result.eligible
    assert result.to_opportunity() is not None


def test_live_signal_refuses_to_invent_edge_without_samples():
    bars = trending_bars(90)
    engine = LiveSignalEngine(
        LiveSignalConfig(
            min_bars=80,
            similarity_band=0.01,
            min_edge_samples=200,
        )
    )
    result = engine.analyze("TEST", bars)

    assert not result.eligible
    assert result.expected_return == 0.0
    assert result.confidence == 0.0
    assert result.to_opportunity() is None


def test_planner_is_idempotent_for_same_closed_bar():
    bars = trending_bars()
    planner = LiveShadowPlanner(
        BarsProvider(bars),
        config=LivePlannerConfig(bar_count=120),
        signal_config=LiveSignalConfig(
            min_bars=80,
            similarity_band=0.20,
            min_edge_samples=12,
        ),
    )

    first = planner.build("TEST")
    second = planner.build("TEST")

    assert first.plan is not None
    assert second.plan is not None
    assert first.plan.decision_id == second.plan.decision_id
    assert first.plan.market_regime == first.analysis.regime.value
    assert first.last_bar_time == bars[-1].timestamp
