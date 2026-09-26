from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from stockbot.data.live_models import AccountSnapshot, MarketBar, MarketQuote, SymbolSpec


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("snapshot timestamps must be timezone-aware")
    return parsed


class SnapshotFileProvider:
    """Read-only provider for snapshots exported by MT5 or another bridge."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _load(self) -> dict:
        if not self.path.exists():
            raise RuntimeError("SNAPSHOT_FILE_MISSING")
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("SNAPSHOT_FILE_INVALID") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("SNAPSHOT_FILE_INVALID")
        return payload

    def account_snapshot(self, *, now: datetime | None = None) -> AccountSnapshot:
        payload = self._load()
        account = payload.get("account")
        if not isinstance(account, dict):
            raise RuntimeError("SNAPSHOT_ACCOUNT_MISSING")
        return AccountSnapshot(
            balance=float(account["balance"]),
            equity=float(account["equity"]),
            free_margin=float(account["free_margin"]),
            timestamp=_parse_time(str(account["timestamp"])),
            currency=str(account.get("currency", "USD")),
        )

    def quote(self, symbol: str) -> MarketQuote:
        payload = self._load()
        symbols = payload.get("symbols")
        if not isinstance(symbols, dict) or symbol not in symbols:
            raise RuntimeError(f"SNAPSHOT_SYMBOL_MISSING:{symbol}")
        quote = symbols[symbol].get("quote")
        if not isinstance(quote, dict):
            raise RuntimeError(f"SNAPSHOT_QUOTE_MISSING:{symbol}")
        return MarketQuote(
            symbol=symbol,
            bid=float(quote["bid"]),
            ask=float(quote["ask"]),
            timestamp=_parse_time(str(quote["timestamp"])),
            source="mt5_snapshot",
        )

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        payload = self._load()
        symbols = payload.get("symbols")
        if not isinstance(symbols, dict) or symbol not in symbols:
            raise RuntimeError(f"SNAPSHOT_SYMBOL_MISSING:{symbol}")
        spec = symbols[symbol].get("spec")
        if not isinstance(spec, dict):
            raise RuntimeError(f"SNAPSHOT_SPEC_MISSING:{symbol}")
        return SymbolSpec(
            symbol=symbol,
            tick_size=float(spec["tick_size"]),
            tick_value=float(spec.get("tick_value", 0.0)),
            volume_min=float(spec["volume_min"]),
            volume_max=float(spec["volume_max"]),
            volume_step=float(spec["volume_step"]),
            trade_enabled=bool(spec.get("trade_enabled", True)),
        )


    def bars(self, symbol: str, timeframe=None, count: int = 250) -> list[MarketBar]:
        if count <= 0:
            raise ValueError("count must be positive")
        payload = self._load()
        symbols = payload.get("symbols")
        if not isinstance(symbols, dict) or symbol not in symbols:
            raise RuntimeError(f"SNAPSHOT_SYMBOL_MISSING:{symbol}")
        rows = symbols[symbol].get("bars")
        if not isinstance(rows, list):
            raise RuntimeError(f"SNAPSHOT_BARS_MISSING:{symbol}")
        selected = rows[-count:]
        return [
            MarketBar(
                symbol=symbol,
                timestamp=_parse_time(str(row["timestamp"])),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
            )
            for row in selected
        ]
