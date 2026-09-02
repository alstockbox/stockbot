from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import quote, urlencode

import pandas as pd

from stockbot.data.market_schema import CANONICAL_BAR_COLUMNS, validate_canonical_bars
from stockbot.data.providers.http import HttpTransport, ProviderError
from stockbot.data.schemas import DataGrade


class TiingoProvider:
    name = "tiingo"
    default_grade = DataGrade.BOOTSTRAP

    def __init__(self, token: str, transport=None) -> None:
        token = str(token or "").strip()
        if not token:
            raise ProviderError("Tiingo token is required")
        self._token = token
        self._transport = transport or HttpTransport()

    @property
    def grade(self) -> DataGrade:
        return self.default_grade

    def fetch_bars(self, symbol: str, start, end) -> pd.DataFrame:
        symbol = str(symbol).strip().upper()
        if not symbol:
            raise ProviderError("symbol is required")
        start_date = pd.Timestamp(start).date().isoformat()
        end_date = pd.Timestamp(end).date().isoformat()
        params = urlencode({"startDate": start_date, "endDate": end_date})
        url = f"https://api.tiingo.com/tiingo/daily/{quote(symbol, safe='')}/prices?{params}"
        payload = self._transport.get_json(
            url,
            {"Authorization": f"Token {self._token}", "Accept": "application/json", "User-Agent": "StockBot/1.2"},
        )
        if not isinstance(payload, list) or not payload:
            raise ProviderError("Tiingo returned no price rows")
        retrieved_at = datetime.now(timezone.utc)
        rows = []
        required = {"date", "open", "high", "low", "close", "volume"}
        for item in payload:
            if not isinstance(item, dict) or not required.issubset(item):
                raise ProviderError("Tiingo returned malformed price row")
            rows.append({
                "timestamp": pd.Timestamp(item["date"]),
                "symbol": symbol,
                "open": item["open"], "high": item["high"], "low": item["low"], "close": item["close"], "volume": item["volume"],
                "adj_open": item.get("adjOpen"), "adj_high": item.get("adjHigh"), "adj_low": item.get("adjLow"),
                "adj_close": item.get("adjClose"), "adj_volume": item.get("adjVolume"),
                "div_cash": item.get("divCash", 0.0) or 0.0,
                "split_factor": item.get("splitFactor", 1.0) or 1.0,
                "provider": self.name,
                "retrieved_at": retrieved_at,
            })
        frame = pd.DataFrame(rows, columns=CANONICAL_BAR_COLUMNS)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["retrieved_at"] = pd.to_datetime(frame["retrieved_at"], utc=True)
        frame = frame.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
        validate_canonical_bars(frame)
        return frame


__all__ = ["ProviderError", "TiingoProvider"]
