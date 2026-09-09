import pandas as pd

from stockbot.research.quarantine import (
    QuarantineConfig,
    assert_same_quarantine_boundary,
    split_sealed_quarantine,
)


def _bars() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=360, freq="B", tz="UTC")
    rows = []
    for symbol in ("AAA", "BBB", "CCC"):
        for i, dt in enumerate(dates):
            price = 100.0 + 0.1 * i
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": price,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def test_sealed_quarantine_never_leaks_rows_back_into_development():
    bars = _bars()
    start = pd.Timestamp("2025-03-03", tz="UTC")
    split = split_sealed_quarantine(
        bars,
        QuarantineConfig(
            start=start.isoformat(),
            min_development_periods=250,
            min_quarantine_periods=40,
        ),
    )

    development_max = pd.to_datetime(split.development_bars["timestamp"], utc=True).max()
    quarantine_min = pd.to_datetime(split.quarantine_bars["timestamp"], utc=True).min()
    assert development_max < start
    assert quarantine_min >= start
    assert len(split.development_bars) + len(split.quarantine_bars) == len(bars)
    assert split.manifest.state == "sealed"
    assert split.manifest.development_rows == len(split.development_bars)
    assert split.manifest.quarantine_rows == len(split.quarantine_bars)


def test_new_future_rows_remain_in_same_sealed_quarantine():
    bars = _bars()
    config = QuarantineConfig(
        start="2025-03-03",
        min_development_periods=250,
        min_quarantine_periods=40,
    )
    first = split_sealed_quarantine(bars, config)
    extra_dates = pd.date_range("2025-05-21", periods=20, freq="B", tz="UTC")
    extra = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "timestamp": dt,
                "open": 150.0,
                "high": 151.0,
                "low": 149.0,
                "close": 150.0,
                "volume": 1_000_000,
            }
            for symbol in ("AAA", "BBB", "CCC")
            for dt in extra_dates
        ]
    )
    second = split_sealed_quarantine(pd.concat([bars, extra], ignore_index=True), config)

    assert len(second.development_bars) == len(first.development_bars)
    assert second.manifest.start == first.manifest.start
    assert second.manifest.quarantine_rows > first.manifest.quarantine_rows


def test_quarantine_identity_changes_when_sealed_market_content_changes():
    bars = _bars()
    config = QuarantineConfig(
        start="2025-03-03",
        min_development_periods=250,
        min_quarantine_periods=40,
    )
    first = split_sealed_quarantine(bars, config)

    changed = bars.copy()
    timestamps = pd.to_datetime(changed["timestamp"], utc=True)
    target = changed.index[(timestamps >= pd.Timestamp("2025-03-03", tz="UTC")) & (changed["symbol"] == "AAA")][0]
    changed.loc[target, "close"] = float(changed.loc[target, "close"]) + 0.01
    second = split_sealed_quarantine(changed, config)

    assert second.manifest.start == first.manifest.start
    assert second.manifest.development_rows == first.manifest.development_rows
    assert second.manifest.quarantine_rows == first.manifest.quarantine_rows
    assert second.manifest.development_periods == first.manifest.development_periods
    assert second.manifest.quarantine_periods == first.manifest.quarantine_periods
    assert second.manifest.quarantine_id != first.manifest.quarantine_id


def test_sealed_boundary_cannot_be_silently_rotated():
    split = split_sealed_quarantine(
        _bars(),
        QuarantineConfig(
            start="2025-03-03",
            min_development_periods=250,
            min_quarantine_periods=40,
        ),
    )
    assert_same_quarantine_boundary(split.manifest, QuarantineConfig(start="2025-03-03"))
    try:
        assert_same_quarantine_boundary(split.manifest, QuarantineConfig(start="2025-04-01"))
    except ValueError as exc:
        assert "sealed" in str(exc)
    else:
        raise AssertionError("expected sealed boundary mismatch to fail")
