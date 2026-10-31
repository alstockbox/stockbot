"""Deterministic walk-forward for XSTO swing RESEARCH (not live or paper).

Every test fold begins flat and with fresh cash, after 200+ indicator-warmup
sessions. Parameter selection sees ONLY bars strictly before the test period.
All results are unverified research hypotheses until historical point-in-time
data, dividends, delistings, benchmark and execution are independently audited.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from datetime import date, datetime, time
from zoneinfo import ZoneInfo
from hashlib import sha256
from math import isfinite
from typing import Any

import pandas as pd

from stockbot.swing.engine import SwingConfig, run_swing_backtest, validate_ohlcv
from stockbot.swing.corporate_actions import CorporateAction, validate_actions
from stockbot.swing.lifecycle import DelistingSettlement, match_settlements
from stockbot.swing.universe import InstrumentHistory, audit_instrument_history
from stockbot.swing.data_quality import _parse_utc


@dataclass(frozen=True)
class WalkForwardConfig:
    train_sessions: int = 504
    test_sessions: int = 126
    min_folds: int = 3
    stop_candidates: tuple[float, ...] = (2.0, 2.5, 3.0)
    stress_execution_bps: tuple[float, ...] = (10.0, 30.0)
    base: SwingConfig = field(default_factory=SwingConfig)

    def __post_init__(self) -> None:
        if self.train_sessions < 252:
            raise ValueError("train_sessions must include indicator warmup plus evaluation")
        if self.test_sessions < 20 or self.min_folds < 2:
            raise ValueError("need at least 20 testing sessions and 2 full folds")
        if not (1 <= len(self.stop_candidates) <= 8):
            raise ValueError("stop_candidates must be a small preregistered grid")
        if (len(set(self.stop_candidates)) != len(self.stop_candidates) or
            any(not isfinite(v) or not 0 < v <= 10 for v in self.stop_candidates)):
            raise ValueError("invalid stop candidate grid")
        if any(not isfinite(v) or v < 0 or v > 500 for v in self.stress_execution_bps):
            raise ValueError("invalid execution cost sensitivity")


@dataclass(frozen=True)
class WalkForwardFold:
    number: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    chosen_atr_stop_multiple: float
    training_net_return_pct: float
    test_net_return_pct: float
    test_drawdown_pct: float
    test_hypothetical_fills: int
    test_commissions_sek: float
    cost_stress_returns_pct: dict[str, float]
    train_universe_symbols: int = 0
    test_universe_symbols: int = 0
    test_lifecycle_events: tuple[str, ...] = ()


@dataclass(frozen=True)
class WalkForwardResult:
    mode: str
    data_quality: str
    data_sha256: str
    independent_benchmark_validated: bool
    money_traded_sek: float
    fold_count: int
    folds: tuple[WalkForwardFold, ...]
    compounded_hypothetical_return_pct: float
    stitched_max_drawdown_pct: float
    total_hypothetical_fills: int
    sufficient_activity_for_live_decision: bool
    source_universe_sha256: str | None = None
    source_settlements_sha256: str | None = None
    source_actions_sha256: str | None = None
    historical_universe_independently_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_walkforward(
    frame: pd.DataFrame,
    config: WalkForwardConfig | None = None,
    *,
    corporate_actions: tuple[CorporateAction, ...] = (),
    universe: tuple[InstrumentHistory, ...] | None = None,
    settlements: tuple[DelistingSettlement, ...] = (),
    collected_at_utc: str | None = None,
) -> WalkForwardResult:
    cfg = config or WalkForwardConfig()
    universe_digest: str | None = None
    if universe is None:
        if settlements:
            raise ValueError("settlements require a historical universe")
        clean = validate_ohlcv(frame)
    else:
        if not collected_at_utc:
            raise ValueError("dynamic walk-forward requires data collection as-of")
        receipt = audit_instrument_history(
            frame, universe, collected_at_utc=collected_at_utc
        )
        universe_digest = receipt.universe_sha256
        clean = frame[list(sorted({"date", "symbol", "open", "high", "low", "close", "volume"}))].copy()
        clean["date"] = pd.to_datetime(clean["date"], errors="raise")
        clean = clean.sort_values(["date", "symbol"]).reset_index(drop=True)
        match_settlements(
            universe, settlements, first_session=clean["date"].min().date(),
            last_session=clean["date"].max().date()
        )
    dates = pd.DatetimeIndex(clean["date"].unique()).sort_values()
    validate_actions(corporate_actions, symbols=set(clean["symbol"]),
                     sessions={d.date() for d in dates})
    needed = cfg.train_sessions + cfg.test_sessions * cfg.min_folds
    if len(dates) < needed:
        raise ValueError(f"insufficient complete sessions for OOS: {len(dates)} < {needed}")
    if cfg.base.first_trade_date is not None:
        raise ValueError("walk-forward owns the first_trade_date cutoff")

    canonical = clean.to_csv(index=False, date_format="%Y-%m-%d", float_format="%.12g").encode()
    fingerprint = sha256(canonical).hexdigest()

    def fingerprint_events(events: tuple) -> str | None:
        if not events:
            return None
        # Include all economic inputs, and make the digest independent
        # of caller order. SHA is tamper-evident, not source authentication.
        json_rows = [
            json.dumps(asdict(event), default=str, sort_keys=True,
                       separators=(",", ":"), ensure_ascii=False)
            for event in events
        ]
        return sha256(("[" + ",".join(sorted(json_rows)) + "]").encode()).hexdigest()

    settlement_digest = fingerprint_events(settlements)
    action_digest = fingerprint_events(corporate_actions)
    folds: list[WalkForwardFold] = []
    combined = [1.0]
    fold_trades = 0

    def scope_records(first: pd.Timestamp, end: pd.Timestamp) -> tuple[InstrumentHistory, ...]:
        if universe is None:
            return ()
        scope = []
        end_asof = datetime.combine(
            end.date(), time(17, 30), tzinfo=ZoneInfo("Europe/Stockholm")
        ).astimezone(ZoneInfo("UTC"))
        for rec in universe:
            if rec.first_session > end.date():
                continue  # future IPO cannot enter current training/test scope
            if rec.last_session is not None and rec.last_session < first.date():
                continue  # delisted before the scope starts
            if rec.last_session is not None and rec.last_session > end.date():
                # Future delisting status/last-date cannot guide training
                # before that delisting has occurred.
                scope.append(replace(
                    rec, status="ACTIVE", last_session=None, last_known_at_utc=None
                ))
            else:
                if rec.last_known_at_utc and _parse_utc(rec.last_known_at_utc) > end_asof:
                    raise ValueError("delisting disclosure after fold cutoff")
                scope.append(rec)
        if not scope:
            raise ValueError("no eligible instruments within fold")
        return tuple(scope)

    for split in range(cfg.train_sessions, len(dates) - cfg.test_sessions + 1, cfg.test_sessions):
        train_start_idx = split - cfg.train_sessions
        train_first = dates[train_start_idx]
        train_end = dates[split - 1]
        test_end = dates[split + cfg.test_sessions - 1]
        train = clean[(clean["date"] >= train_first) & (clean["date"] <= train_end)]
        through_test = clean[clean["date"] <= test_end]
        train_records = scope_records(train_first, train_end)
        test_records = scope_records(dates[0], test_end)
        train_symbols = set(train["symbol"])
        test_symbols = set(through_test["symbol"])
        # Events after each cutoff or outside that scope cannot enter a fold.
        train_actions = tuple(
            e for e in corporate_actions
            if train_first.date() <= e.ex_date <= train_end.date()
            and e.symbol in train_symbols
        )
        test_actions = tuple(
            e for e in corporate_actions
            if e.ex_date <= test_end.date() and e.symbol in test_symbols
        )
        train_settlements = tuple(
            e for e in settlements
            if train_first.date() <= e.last_session <= train_end.date()
        )
        test_settlements = tuple(
            e for e in settlements if e.last_session <= test_end.date()
        )
        training_trade_start = dates[train_start_idx + 200].strftime("%Y-%m-%d")
        test_trade_start = dates[split].strftime("%Y-%m-%d")
        # Candidate tuning uses TRAIN window only; no test P&L in selection.
        candidates: list[tuple[float, float, float]] = []
        for multiplier in cfg.stop_candidates:
            training_cfg = replace(cfg.base,
                                   atr_stop_multiple=multiplier,
                                   first_trade_date=training_trade_start)
            training = run_swing_backtest(
                train, training_cfg, corporate_actions=train_actions,
                universe=train_records if universe is not None else None,
                settlements=train_settlements,
                collected_at_utc=collected_at_utc,
            )
            net_return = training.final_equity_sek / cfg.base.initial_sek - 1.0
            # Penalize drawdown; don't optimize on test results.
            score = net_return - 1.5 * training.max_drawdown
            candidates.append((score, -multiplier, net_return))
        score, negative_multiplier, net_training_return = max(candidates)
        chosen = -negative_multiplier

        test_cfg = replace(cfg.base, atr_stop_multiple=chosen,
                           first_trade_date=test_trade_start)
        tested = run_swing_backtest(
            through_test, test_cfg, corporate_actions=test_actions,
            universe=test_records if universe is not None else None,
            settlements=test_settlements,
            collected_at_utc=collected_at_utc,
        )
        oos_slice = tested.equity.loc[tested.equity.index >= dates[split]]
        # No trades before first test date and no opening positions carried from
        # a synthetic training account: fold starts with initial capital.
        assert oos_slice.size == cfg.test_sessions
        net = tested.final_equity_sek / cfg.base.initial_sek - 1.0
        # Stitch fold net-value path, preserving compounding across OOS folds.
        fold_start_value = combined[-1]
        for value in oos_slice.to_numpy():
            combined.append(fold_start_value * float(value) / cfg.base.initial_sek)

        sensitivities: dict[str, float] = {}
        for cost_bps in cfg.stress_execution_bps:
            stressed = run_swing_backtest(
                through_test, replace(test_cfg, execution_bps_each_side=cost_bps),
                corporate_actions=test_actions,
                universe=test_records if universe is not None else None,
                settlements=test_settlements,
                collected_at_utc=collected_at_utc,
            )
            sensitivities[f"{cost_bps:g}bps"] = round(
                (stressed.final_equity_sek / cfg.base.initial_sek - 1.0) * 100, 6
            )
        fold_trades += len(tested.fills)
        folds.append(WalkForwardFold(
            number=len(folds) + 1,
            train_start=train_first.strftime("%Y-%m-%d"),
            train_end=train_end.strftime("%Y-%m-%d"),
            test_start=test_trade_start,
            test_end=dates[split + cfg.test_sessions - 1].strftime("%Y-%m-%d"),
            chosen_atr_stop_multiple=chosen,
            training_net_return_pct=round(net_training_return * 100, 6),
            test_net_return_pct=round(net * 100, 6),
            test_drawdown_pct=round(tested.max_drawdown * 100, 6),
            test_hypothetical_fills=len(tested.fills),
            test_commissions_sek=round(tested.total_fees_sek, 4),
            cost_stress_returns_pct=sensitivities,
            train_universe_symbols=len(train_records) if universe is not None else len(train_symbols),
            test_universe_symbols=len(test_records) if universe is not None else len(test_symbols),
            test_lifecycle_events=tuple(
                event for event in tested.lifecycle_events
                if event[:10] >= test_trade_start
            ),
        ))

    equity = pd.Series(combined)
    peak = equity.cummax()
    drawdown = 1 - equity / peak
    return WalkForwardResult(
        mode="WALK_FORWARD_BACKTEST_ONLY",
        data_quality="UNVERIFIED",
        data_sha256=fingerprint,
        independent_benchmark_validated=False,
        money_traded_sek=0.0,
        fold_count=len(folds),
        folds=tuple(folds),
        compounded_hypothetical_return_pct=round((equity.iloc[-1] - 1.0) * 100, 6),
        stitched_max_drawdown_pct=round(float(drawdown.max()) * 100, 6),
        total_hypothetical_fills=fold_trades,
        sufficient_activity_for_live_decision=False,
        source_universe_sha256=universe_digest,
        source_settlements_sha256=settlement_digest,
        source_actions_sha256=action_digest,
        historical_universe_independently_verified=False,
    )
