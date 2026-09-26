from datetime import datetime, timedelta, timezone
import json

from stockbot.brokers.mt5_readonly import MT5ReadOnlyAdapter
from stockbot.brokers.snapshot_feed import SnapshotFileProvider
from stockbot.data.live_models import AccountSnapshot, MarketQuote, SymbolSpec
from stockbot.data.live_validation import (
    LiveValidationConfig,
    validate_account,
    validate_quote,
    validate_symbol,
)


class FakeMT5:
    def __init__(self, now: datetime):
        self.now = now

    def account_info(self):
        return {
            "balance": 466.0,
            "equity": 470.0,
            "margin_free": 450.0,
            "currency": "USD",
        }

    def symbol_info(self, symbol):
        return {
            "trade_tick_size": 0.00001,
            "trade_tick_value": 1.0,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "trade_mode": 1,
        }

    def symbol_info_tick(self, symbol):
        return {
            "bid": 1.1000,
            "ask": 1.1002,
            "time_msc": int(self.now.timestamp() * 1000),
        }

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        return [
            {
                "time": int(self.now.timestamp()),
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "tick_volume": 100,
            }
        ]

    def order_send(self, request):
        raise AssertionError("must never be reachable through adapter")


def test_mt5_adapter_exposes_read_only_market_state():
    now = datetime(2026, 9, 26, tzinfo=timezone.utc)
    adapter = MT5ReadOnlyAdapter(FakeMT5(now))

    account = adapter.account_snapshot(now=now)
    quote = adapter.quote("EURUSD")
    spec = adapter.symbol_spec("EURUSD")
    bars = adapter.bars("EURUSD", timeframe=1, count=1)

    assert account.equity == 470.0
    assert quote.ask == 1.1002
    assert spec.volume_min == 0.01
    assert len(bars) == 1
    assert not hasattr(adapter, "order_send")
    assert not hasattr(adapter, "order_check")


def test_live_validation_rejects_stale_and_excessive_spread():
    now = datetime(2026, 9, 26, tzinfo=timezone.utc)
    config = LiveValidationConfig(max_quote_age_seconds=30.0, max_spread_fraction=0.02)

    stale = MarketQuote(
        "EURUSD",
        1.1000,
        1.1002,
        now - timedelta(seconds=31),
        "test",
    )
    wide = MarketQuote("EURUSD", 1.00, 1.05, now, "test")

    assert validate_quote(stale, now=now, config=config).reasons == ("STALE_QUOTE",)
    assert "EXCESSIVE_SPREAD" in validate_quote(wide, now=now, config=config).reasons


def test_account_and_symbol_validation_are_fail_closed():
    now = datetime(2026, 9, 26, tzinfo=timezone.utc)
    account = AccountSnapshot(466.0, 466.0, 466.0, now, "USD")
    spec = SymbolSpec("EURUSD", 0.00001, 1.0, 0.01, 100.0, 0.01, False)

    assert validate_account(account, now=now).ok
    assert validate_symbol(spec).reasons == ("SYMBOL_DISABLED",)


def test_snapshot_provider_reads_export_format(tmp_path):
    now = datetime(2026, 9, 26, tzinfo=timezone.utc)
    payload = {
        "account": {
            "balance": 466.0,
            "equity": 466.0,
            "free_margin": 450.0,
            "currency": "USD",
            "timestamp": now.isoformat(),
        },
        "symbols": {
            "EURUSD": {
                "quote": {
                    "bid": 1.1,
                    "ask": 1.1002,
                    "timestamp": now.isoformat(),
                },
                "spec": {
                    "tick_size": 0.00001,
                    "tick_value": 1.0,
                    "volume_min": 0.01,
                    "volume_max": 100.0,
                    "volume_step": 0.01,
                    "trade_enabled": True,
                },
            }
        },
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    provider = SnapshotFileProvider(path)
    assert provider.account_snapshot().equity == 466.0
    assert provider.quote("EURUSD").bid == 1.1
    assert provider.symbol_spec("EURUSD").trade_enabled
