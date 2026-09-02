from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import quote, urlencode

import pandas as pd

from stockbot.data.market_schema import CANONICAL_BAR_COLUMNS, validate_canonical_bars
from stockbot.data.providers.http import HttpTransport, ProviderError
from stockbot.data.schemas import DataGrade


class YahooBootstrapProvider:
    name = "yahoo-bootstrap"

    def __init__(self, transport=None) -> None:
        self._transport = transport or HttpTransport()

    @property
    def grade(self) -> DataGrade:
        return DataGrade.BOOTSTRAP

    def fetch_bars(self, symbol: str, start, end) -> pd.DataFrame:
        symbol = str(symbol).strip().upper()
        if not symbol:
            raise ProviderError("symbol is required")
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
        params = urlencode({
            "period1": int(start_ts.timestamp()),
            "period2": int((end_ts + pd.Timedelta(days=1)).timestamp()),
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        })
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}?{params}"
        payload = self._transport.get_json(url, {"User-Agent": "Mozilla/5.0 StockBot/1.2", "Accept": "application/json"})
        try:
            result = payload["chart"]["result"]
            if not result:
                raise KeyError("empty result")
            result = result[0]
            timestamps = result["timestamp"]
            quote_data = result["indicators"]["quote"][0]
        except (TypeError, KeyError, IndexError) as exc:
            raise ProviderError("Yahoo bootstrap returned malformed chart data") from exc
        if not timestamps:
            raise ProviderError("Yahoo bootstrap returned no timestamps")

        adj_values = None
        try:
            adj_values = result["indicators"]["adjclose"][0]["adjclose"]
        except (TypeError, KeyError, IndexError):
            pass
        events = result.get("events") or {}
        dividends = events.get("dividends") or {}
        splits = events.get("splits") or {}
        retrieved_at = datetime.now(timezone.utc)
        rows = []
        for i, epoch in enumerate(timestamps):
            try:
                open_ = quote_data["open"][i]
                high = quote_data["high"][i]
                low = quote_data["low"][i]
                close = quote_data["close"][i]
                volume = quote_data["volume"][i]
            except (KeyError, IndexError, TypeError) as exc:
                raise ProviderError("Yahoo bootstrap quote arrays are malformed") from exc
            if any(value is None for value in (open_, high, low, close, volume)):
                continue
            dividend = dividends.get(str(epoch)) or dividends.get(epoch) or {}
            split = splits.get(str(epoch)) or splits.get(epoch) or {}
            numerator = split.get("numerator")
            denominator = split.get("denominator")
            split_factor = 1.0
            if numerator not in (None, 0) and denominator not in (None, 0):
                split_factor = float(numerator) / float(denominator)
            elif split.get("splitRatio"):
                ratio = str(split["splitRatio"])
                if ":" in ratio:
                    left, right = ratio.split(":", 1)
                    split_factor = float(left) / float(right)
            adj_close = adj_values[i] if adj_values is not None and i < len(adj_values) else None
            rows.append({
                "timestamp": pd.to_datetime(epoch, unit="s", utc=True),
                "symbol": symbol,
                "open": open_, "high": high, "low": low, "close": close, "volume": volume,
                "adj_open": None, "adj_high": None, "adj_low": None, "adj_close": adj_close, "adj_volume": None,
                "div_cash": float(dividend.get("amount", 0.0) or 0.0),
                "split_factor": split_factor,
                "provider": self.name,
                "retrieved_at": retrieved_at,
            })
        if not rows:
            raise ProviderError("Yahoo bootstrap returned no usable rows")
        frame = pd.DataFrame(rows, columns=CANONICAL_BAR_COLUMNS)
        frame = frame.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
        validate_canonical_bars(frame)
        return frame


__all__ = ["ProviderError", "YahooBootstrapProvider"]
