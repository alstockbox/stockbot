"""Deterministic SYNTHETIC ONLY EOD signal candidate, not a real paper bot.

No external data, no broker, no scheduler. Default preview is read-only.
Optional stage delegates to the existing hash-bound synthetic workflow.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, date
from math import floor, isfinite

import pandas as pd
import exchange_calendars as xcals

from stockbot.swing.engine import SwingConfig, _indicators
from stockbot.swing.paper_feed import _parse_utc, _utc
from stockbot.swing.paper_rehearsal import SyntheticPaperWorkflow


@dataclass(frozen=True)
class SignalPreview:
    mode: str
    status: str
    source_session: str | None
    source_entry_sha256: str | None
    symbol: str | None
    side: str | None
    quantity: int
    signal_reason: str
    observed_at_utc: str | None
    last_close_sek: float | None
    atr_sek: float | None
    estimated_equity_sek: float | None
    estimated_risk_sek: float
    position_limit_sek: float
    technical_sessions: int
    staged_decision_id: str | None = None
    authorized_real_market_data: bool = False
    actual_forward_paper_started: bool = False
    broker_orders_sent: int = 0

    def to_dict(self):
        return asdict(self)


class SyntheticSwingSignal:
    """Rule-locked single-decision-per-session EOD synthetic gate.

    Uses only finalized and observed source candles. Chooses existing trend
    exits before breakout/pullback entries. Has no intraday stop service.
    """

    def __init__(self, workflow: SyntheticPaperWorkflow, *,
                 config: SwingConfig | None = None):
        self.workflow = workflow
        self.cfg = config or SwingConfig(initial_sek=workflow.ledger.initial_cash_sek)
        if self.cfg.initial_sek != workflow.ledger.initial_cash_sek:
            raise ValueError("STRATEGY_AND_LEDGER_INITIAL_CASH_MISMATCH")
        # Must not pretend that supported alpha or live readiness is proven.
        self.strategy_id = "SWING_CLOSE_SIGNAL_SYNTHETIC_V1"

    def preview(self, *, clock_utc: datetime) -> SignalPreview:
        now = _utc(clock_utc)
        feed = self.workflow._verified_feed()
        if not feed:
            raise ValueError("no synthetic feed observations")
        days = sorted(feed)
        latest = days[-1]
        source = feed[latest]
        observed = _parse_utc(source["session"]["observed_at_utc"])
        # No future-dated historical batch may enter an as-of calculation.
        if any(_parse_utc(row["session"]["observed_at_utc"]) > now for row in feed.values()):
            raise ValueError("FUTURE_SYNTHETIC_FEED_NOT_AVAILABLE_AT_DECISION")
        balance = self.workflow.ledger.snapshot()
        latest_bars = {x["symbol"]: x for x in source["session"]["bars"]}
        if any(sym not in latest_bars or latest_bars[sym]["close"] <= 0
               for sym in balance.positions):
            raise ValueError("UNMARKED_HELD_SYMBOL_BLOCKED")
        equity = balance.cash_sek + sum(
            qty * float(latest_bars[sym]["close"])
            for sym,qty in balance.positions.items()
        )
        base = dict(
            mode=self.strategy_id,
            source_session=latest.isoformat(),
            source_entry_sha256=source["entry_hash"],
            observed_at_utc=observed.isoformat(),
            estimated_equity_sek=round(equity, 6),
            position_limit_sek=round(equity * self.cfg.max_position_weight, 6),
            technical_sessions=len(days),
        )

        def result(status: str, reason: str, *, symbol: str | None = None,
                   side: str | None = None, quantity: int = 0,
                   close: float | None = None, atr: float | None = None,
                   risk: float = 0.0) -> SignalPreview:
            return SignalPreview(
                status=status, signal_reason=reason, symbol=symbol,
                side=side, quantity=quantity,
                last_close_sek=close, atr_sek=atr,
                estimated_risk_sek=round(risk, 6), **base
            )

        # An old bar can be analyzed for research, but must NOT be
        # presented as an actionable next-open signal after that open.
        # Keep it fail-closed: no retroactive creation of new decisions.
        calendar = xcals.get_calendar("XSTO")
        next_open = calendar.session_open(
            calendar.next_session(pd.Timestamp(latest))
        ).to_pydatetime()
        if now >= next_open:
            return result("EXPIRED_SOURCE", "Next XSTO opening has already passed")
        if len(days) < 205:
            return result("INSUFFICIENT_HISTORY", "At least 205 contiguous XSTO sessions required")
        first_symbols = {b["symbol"] for b in feed[days[0]]["session"]["bars"]}
        if any({b["symbol"] for b in feed[day]["session"]["bars"]} != first_symbols
               for day in days):
            raise ValueError("DYNAMIC_UNIVERSE_NOT_SUPPORTED_IN_AUTOSIGNAL")
        if len(balance.positions) > self.cfg.max_positions:
            raise ValueError("TOO_MANY_OPEN_POSITIONS_FOR_STRATEGY")

        series_by_symbol = {}
        for symbol in sorted(first_symbols):
            rows = []
            for day in days:
                bar = self.workflow._find_bar(feed[day], symbol)
                rows.append({
                    "date": pd.Timestamp(day), **bar,
                })
            raw = pd.DataFrame(rows)
            indicators = _indicators(raw)
            # Indicator values include this session and earlier only.
            series_by_symbol[symbol] = (raw, indicators.iloc[-1])

        # Any held stock with last-close exit condition is prioritized
        # over new buys. EXIT order is a next-open intention, not a stop fill.
        for symbol in sorted(balance.positions):
            raw, ind = series_by_symbol[symbol]
            close = float(ind["close"])
            atr = float(ind["atr"])
            if not isfinite(atr) or atr <= 0:
                raise ValueError("HELD_SYMBOL_WITHOUT_VALID_ATR")
            if bool(ind["exit"]):
                return result("SIGNAL", "CLOSE_BELOW_SMA50_TREND_EXIT",
                              symbol=symbol, side="SELL",
                              quantity=balance.positions[symbol],
                              close=close, atr=atr)
        if equity <= self.cfg.initial_sek * (1 - self.cfg.max_drawdown):
            return result("BLOCKED", "INITIAL_CAPITAL_LOSS_KILL_SWITCH")
        if len(balance.positions) >= self.cfg.max_positions:
            return result("NO_SIGNAL", "MAX_OPEN_POSITIONS")

        # Deterministic ranking matches swing engine: trend strength vs
        # SMA50, ticker lexical tie-break. No future session OHLC used.
        candidates = []
        for symbol, (raw, ind) in series_by_symbol.items():
            if symbol in balance.positions:
                continue
            if raw.iloc[-1]["volume"] <= 0 or not str(ind["entry"]):
                continue
            close = float(ind["close"])
            sma50 = float(ind["sma50"])
            atr = float(ind["atr"])
            if not all(isfinite(v) and v > 0 for v in (close, sma50, atr)):
                continue
            prior_vol = raw["volume"].tail(20).median()
            median_turnover = float(raw["volume"].tail(20).mul(raw["close"].tail(20)).median())
            if not isfinite(median_turnover) or median_turnover < self.cfg.min_prior_median_turnover_sek:
                continue
            forecast_cap = floor(float(prior_vol) * self.cfg.max_daily_volume_fraction)
            stop_distance = self.cfg.atr_stop_multiple * atr
            if stop_distance >= close or forecast_cap < 1:
                continue
            # Today's close is a sizing proxy, *not* guaranteed tomorrow's
            # open; actual synthetic reconciliation blocks overspending.
            budget = min(equity*self.cfg.max_position_weight,
                         equity*self.cfg.risk_per_trade/stop_distance*close)
            qty = min(floor(budget/close), forecast_cap)
            reference_stop = close - stop_distance
            while qty > 0:
                notional = qty*close
                buy_fee = max(self.cfg.minimum_commission_sek,
                              self.cfg.commission_rate*notional)
                exit_fee = max(self.cfg.minimum_commission_sek,
                               self.cfg.commission_rate*qty*reference_stop)
                planned_loss_with_fees = qty*stop_distance + buy_fee + exit_fee
                if (notional+buy_fee <= balance.cash_sek + 1e-8
                    and planned_loss_with_fees <=
                        equity*self.cfg.risk_per_trade + 1e-8):
                    break
                qty -= 1
            if qty < 1:
                continue
            candidates.append((-float(close/sma50-1.0),symbol,qty,
                               close,atr,planned_loss_with_fees,str(ind["entry"])))
        if not candidates:
            return result("NO_SIGNAL", "NO_LIQUID_RISK_BOUNDED_ENTRY")
        _,symbol,qty,close,atr,risk,reason = min(candidates)
        return result("SIGNAL", reason, symbol=symbol, side="BUY",
                      quantity=qty, close=close, atr=atr, risk=risk)

    def stage(self, *, clock_utc: datetime) -> SignalPreview:
        """Explicit synthetic-only operation. No autonomous schedule or broker."""
        preview = self.preview(clock_utc=clock_utc)
        if preview.status != "SIGNAL":
            return preview
        decision = self.workflow.plan(
            symbol=preview.symbol, side=preview.side,
            quantity=preview.quantity, clock_utc=clock_utc,
            commission_pct=self.cfg.commission_rate,
            min_commission_sek=self.cfg.minimum_commission_sek,
            expected_source_entry_sha256=preview.source_entry_sha256,
            expected_source_session=date.fromisoformat(preview.source_session),
            max_buy_notional_sek=(
                preview.position_limit_sek if preview.side == "BUY" else None
            ),
            max_buy_risk_sek=(
                round(preview.estimated_equity_sek * self.cfg.risk_per_trade, 6)
                if preview.side == "BUY" else None
            ),
            reference_stop_sek=(
                preview.last_close_sek - self.cfg.atr_stop_multiple * preview.atr_sek
                if preview.side == "BUY" else None
            ),
        )
        return SignalPreview(**{
            **preview.to_dict(), "staged_decision_id":decision,
        })
