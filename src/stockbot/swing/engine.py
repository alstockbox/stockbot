"""Causal, broker-isolated Swedish equity swing research engine.

No broker/network code. All prices are SEK and positions are long-only.
Daily signals use close(t); fills cannot occur earlier than open(t+1).
Synthetic or historical backtests are not live-performance evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Literal

import pandas as pd

REQUIRED = {"date", "symbol", "open", "high", "low", "close", "volume"}


@dataclass(frozen=True)
class SwingConfig:
    initial_sek: float = 1_000.0
    risk_per_trade: float = 0.005
    max_position_weight: float = 0.20
    max_positions: int = 5
    max_drawdown: float = 0.08
    commission_rate: float = 0.0025
    minimum_commission_sek: float = 1.0
    execution_bps_each_side: float = 10.0
    atr_stop_multiple: float = 2.5

    def __post_init__(self) -> None:
        if not isfinite(self.initial_sek) or self.initial_sek <= 0:
            raise ValueError("initial_sek must be positive")
        if not (0 < self.risk_per_trade <= 0.02):
            raise ValueError("risk_per_trade outside research limits")
        if not (0 < self.max_position_weight <= 1):
            raise ValueError("invalid max_position_weight")
        if not (1 <= self.max_positions <= 100):
            raise ValueError("invalid max_positions")
        if not (0 < self.max_drawdown < 1):
            raise ValueError("invalid max_drawdown")
        if not (0 <= self.commission_rate <= 0.1):
            raise ValueError("invalid commission_rate")
        if not (0 <= self.minimum_commission_sek < self.initial_sek):
            raise ValueError("invalid minimum_commission_sek")
        if not (0 <= self.execution_bps_each_side <= 500):
            raise ValueError("invalid execution_bps_each_side")
        if not (0 < self.atr_stop_multiple <= 10):
            raise ValueError("invalid atr_stop_multiple")


@dataclass(frozen=True)
class Fill:
    date: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    executed_price: float
    fee_sek: float
    reason: str


@dataclass(frozen=True)
class SwingResult:
    equity: pd.Series
    fills: tuple[Fill, ...]
    final_equity_sek: float
    max_drawdown: float
    total_fees_sek: float
    total_execution_impact_sek: float


@dataclass
class _Position:
    quantity: int
    stop: float


def _fee(value: float, cfg: SwingConfig) -> float:
    return max(cfg.minimum_commission_sek, cfg.commission_rate * value)


def _execution_price(raw: float, buy: bool, cfg: SwingConfig) -> float:
    impact = cfg.execution_bps_each_side / 10_000
    return raw * (1 + impact if buy else 1 - impact)


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    df = frame[list(sorted(REQUIRED))].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df["date"].isna().any() or (df["date"] != df["date"].dt.normalize()).any():
        raise ValueError("dates must be valid daily timestamps")
    df["symbol"] = df["symbol"].astype(str)
    if df["symbol"].str.fullmatch(r"[A-Za-z0-9._-]+").eq(False).any():
        raise ValueError("invalid symbol")
    for field in ("open", "high", "low", "close", "volume"):
        df[field] = pd.to_numeric(df[field], errors="raise")
        if not df[field].map(isfinite).all():
            raise ValueError(f"non-finite {field}")
    if df.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate symbol/date")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("prices must be positive")
    if (df["volume"] < 0).any():
        raise ValueError("negative volume")
    if ((df["high"] < df[["open", "close", "low"]].max(axis=1)) |
            (df["low"] > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("invalid OHLC range")
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    dates = df["date"].unique()
    if len(dates) < 205:
        raise ValueError("at least 205 synchronized sessions required")
    counts = df.groupby("symbol")["date"].nunique()
    if not counts.eq(len(dates)).all():
        raise ValueError("all symbols must have identical session dates; missing bars fail closed")
    return df


def _indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.set_index("date").sort_index().copy()
    c = df["close"]
    df["sma20"] = c.rolling(20, min_periods=20).mean()
    df["sma50"] = c.rolling(50, min_periods=50).mean()
    df["sma200"] = c.rolling(200, min_periods=200).mean()
    prev = c.shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=14).mean()
    breakout = c > df["high"].shift(1).rolling(20, min_periods=20).max()
    trend = (c > df["sma50"]) & (df["sma50"] > df["sma200"])
    pullback = (
        (c > df["sma20"]) &
        (prev <= df["sma20"].shift(1)) &
        (c > df["sma200"]) & trend
    )
    df["entry"] = ""
    df.loc[trend & breakout, "entry"] = "BREAKOUT"
    df.loc[trend & pullback & ~breakout, "entry"] = "PULLBACK"
    df["exit"] = c < df["sma50"]
    return df


def run_swing_backtest(frame: pd.DataFrame, cfg: SwingConfig | None = None) -> SwingResult:
    """Simulate next-open entries/exits, gap-aware stop exits, and whole-share fills.

    Signal at previous CLOSE; next OPEN execution. Stop checks use the current
    day's low, but stops are based only on previous closes and ATR. This is
    approximate OHLC execution, not proof of executable intraday fills.
    """
    cfg = cfg or SwingConfig()
    clean = validate_ohlcv(frame)
    panels = {
        sym: _indicators(group)
        for sym, group in clean.groupby("symbol", sort=True)
    }
    dates = list(next(iter(panels.values())).index)
    cash = cfg.initial_sek
    positions: dict[str, _Position] = {}
    history: list[float] = []
    fills: list[Fill] = []
    fees = 0.0
    impact_sek = 0.0
    peak = cfg.initial_sek
    halted = False

    for i, today in enumerate(dates):
        previous_day = dates[i - 1] if i else None

        def sell(symbol: str, raw_price: float, reason: str) -> None:
            nonlocal cash, fees, impact_sek
            pos = positions.pop(symbol)
            executed = _execution_price(raw_price, False, cfg)
            value = executed * pos.quantity
            charge = _fee(value, cfg)
            cash += value - charge
            fees += charge
            impact_sek += (raw_price - executed) * pos.quantity
            fills.append(Fill(today.strftime("%Y-%m-%d"), symbol, "SELL",
                              pos.quantity, executed, charge, reason))

        # Exit existing positions first: overnight gaps fill at OPEN, not STOP.
        for symbol in sorted(tuple(positions)):
            row = panels[symbol].loc[today]
            pos = positions[symbol]
            if float(row["open"]) <= pos.stop:
                sell(symbol, float(row["open"]), "GAP_STOP")
            elif previous_day is not None and bool(panels[symbol].loc[previous_day, "exit"]):
                sell(symbol, float(row["open"]), "TREND_EXIT")
            elif float(row["low"]) <= pos.stop:
                sell(symbol, pos.stop, "INTRADAY_STOP")

        open_equity = cash + sum(
            p.quantity * float(panels[s].loc[today, "open"])
            for s, p in positions.items()
        )
        peak = max(peak, open_equity)
        if open_equity <= peak * (1 - cfg.max_drawdown):
            halted = True  # latch: restart requires explicit review

        if previous_day is not None and not halted:
            for symbol in sorted(panels):
                if symbol in positions or len(positions) >= cfg.max_positions:
                    continue
                prior = panels[symbol].loc[previous_day]
                strategy = str(prior["entry"])
                if not strategy:
                    continue
                atr = float(prior["atr"])
                if not isfinite(atr) or atr <= 0:
                    continue
                raw = float(panels[symbol].loc[today, "open"])
                executed = _execution_price(raw, True, cfg)
                stop_distance = cfg.atr_stop_multiple * atr
                if stop_distance <= 0 or stop_distance >= executed:
                    continue
                budget = min(open_equity * cfg.max_position_weight,
                             open_equity * cfg.risk_per_trade / stop_distance * executed)
                quantity = floor(budget / executed)
                while quantity > 0 and quantity * executed + _fee(quantity * executed, cfg) > cash + 1e-9:
                    quantity -= 1
                if quantity < 1:
                    continue
                notional = quantity * executed
                charge = _fee(notional, cfg)
                cash -= notional + charge
                fees += charge
                impact_sek += (executed - raw) * quantity
                positions[symbol] = _Position(quantity, executed - stop_distance)
                fills.append(Fill(today.strftime("%Y-%m-%d"), symbol, "BUY",
                                  quantity, executed, charge, strategy))
                # A stop may be hit on the entry day. OHLC ordering is unknown;
                # conservatively assume the stop triggers if the daily low crossed it.
                if float(panels[symbol].loc[today, "low"]) <= positions[symbol].stop:
                    sell(symbol, positions[symbol].stop, "ENTRY_DAY_STOP")

        equity = cash + sum(
            pos.quantity * float(panels[sym].loc[today, "close"])
            for sym, pos in positions.items()
        )
        peak = max(peak, equity)
        if equity <= peak * (1 - cfg.max_drawdown):
            halted = True
        history.append(equity)

        # Trailing stop updates apply FROM NEXT SESSION (no same-bar lookahead).
        for symbol, pos in positions.items():
            row = panels[symbol].loc[today]
            atr = float(row["atr"])
            if isfinite(atr) and atr > 0:
                pos.stop = max(pos.stop, float(row["close"]) - cfg.atr_stop_multiple * atr)

    equity_series = pd.Series(history, index=pd.DatetimeIndex(dates), name="equity_sek")
    dd = 1 - equity_series / equity_series.cummax().clip(lower=1e-12)
    return SwingResult(equity_series, tuple(fills), float(equity_series.iloc[-1]),
                       float(dd.max()), fees, impact_sek)
