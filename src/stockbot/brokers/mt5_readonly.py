from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from stockbot.data.live_models import AccountSnapshot, MarketBar, MarketQuote, SymbolSpec


@runtime_checkable
class MT5ReadBackend(Protocol):
    def account_info(self) -> Any: ...
    def symbol_info(self, symbol: str) -> Any: ...
    def symbol_info_tick(self, symbol: str) -> Any: ...
    def copy_rates_from_pos(
        self,
        symbol: str,
        timeframe: Any,
        start_pos: int,
        count: int,
    ) -> Any: ...


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _timestamp_from_tick(tick: Any) -> datetime:
    millis = _value(tick, "time_msc")
    if millis not in (None, 0):
        return datetime.fromtimestamp(float(millis) / 1000.0, tz=timezone.utc)
    seconds = _value(tick, "time")
    if seconds in (None, 0):
        raise ValueError("MT5 tick has no timestamp")
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc)


class MT5ReadOnlyAdapter:
    """Read-only facade over an MT5-like backend.

    The adapter deliberately exposes no order_send/order_check/trading methods.
    """

    def __init__(self, backend: MT5ReadBackend, *, source: str = "mt5") -> None:
        self._backend = backend
        self.source = source

    def account_snapshot(self, *, now: datetime | None = None) -> AccountSnapshot:
        info = self._backend.account_info()
        if info is None:
            raise RuntimeError("MT5_ACCOUNT_UNAVAILABLE")
        timestamp = now or datetime.now(timezone.utc)
        free_margin = _value(info, "margin_free", _value(info, "free_margin", 0.0))
        return AccountSnapshot(
            balance=float(_value(info, "balance")),
            equity=float(_value(info, "equity")),
            free_margin=float(free_margin),
            timestamp=timestamp,
            currency=str(_value(info, "currency", "USD")),
        )

    def quote(self, symbol: str) -> MarketQuote:
        tick = self._backend.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"MT5_QUOTE_UNAVAILABLE:{symbol}")
        return MarketQuote(
            symbol=symbol,
            bid=float(_value(tick, "bid")),
            ask=float(_value(tick, "ask")),
            timestamp=_timestamp_from_tick(tick),
            source=self.source,
        )

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        info = self._backend.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"MT5_SYMBOL_UNAVAILABLE:{symbol}")
        trade_mode = int(_value(info, "trade_mode", 1))
        return SymbolSpec(
            symbol=symbol,
            tick_size=float(
                _value(info, "trade_tick_size", _value(info, "point", 0.0))
            ),
            tick_value=float(_value(info, "trade_tick_value", 0.0)),
            volume_min=float(_value(info, "volume_min", 0.0)),
            volume_max=float(_value(info, "volume_max", 0.0)),
            volume_step=float(_value(info, "volume_step", 0.0)),
            trade_enabled=trade_mode != 0,
        )

    def bars(self, symbol: str, timeframe: Any, count: int) -> list[MarketBar]:
        if count <= 0:
            raise ValueError("count must be positive")
        rows = self._backend.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rows is None:
            raise RuntimeError(f"MT5_BARS_UNAVAILABLE:{symbol}")
        output: list[MarketBar] = []
        for row in rows:
            output.append(
                MarketBar(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(
                        float(_value(row, "time")),
                        tz=timezone.utc,
                    ),
                    open=float(_value(row, "open")),
                    high=float(_value(row, "high")),
                    low=float(_value(row, "low")),
                    close=float(_value(row, "close")),
                    volume=float(
                        _value(
                            row,
                            "tick_volume",
                            _value(row, "real_volume", 0.0),
                        )
                    ),
                )
            )
        return output

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"{type(self).__name__} is read-only; backend attribute {name!r} is not exposed"
        )
