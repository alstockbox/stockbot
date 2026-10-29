"""Deterministic walk-forward for XSTO swing RESEARCH (not live or paper).

Every test fold begins flat and with fresh cash, after 200+ indicator-warmup
sessions. Parameter selection sees ONLY bars strictly before the test period.
All results are unverified research hypotheses until historical point-in-time
data, dividends, delistings, benchmark and execution are independently audited.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from math import isfinite
from typing import Any

import pandas as pd

from stockbot.swing.engine import SwingConfig, run_swing_backtest, validate_ohlcv
from stockbot.swing.corporate_actions import CorporateAction, validate_actions


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_walkforward(frame: pd.DataFrame, config: WalkForwardConfig | None = None, *,
                    corporate_actions: tuple[CorporateAction, ...] = ()) -> WalkForwardResult:
    cfg = config or WalkForwardConfig()
    clean = validate_ohlcv(frame)
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
    folds: list[WalkForwardFold] = []
    combined = [1.0]
    fold_trades = 0

    for split in range(cfg.train_sessions, len(dates) - cfg.test_sessions + 1, cfg.test_sessions):
        train = clean[clean["date"] <= dates[split - 1]]
        through_test = clean[clean["date"] <= dates[split + cfg.test_sessions - 1]]
        train_actions = tuple(e for e in corporate_actions if e.ex_date <= dates[split - 1].date())
        test_actions = tuple(e for e in corporate_actions if e.ex_date <= dates[split + cfg.test_sessions - 1].date())
        training_trade_start = dates[200].strftime("%Y-%m-%d")
        test_trade_start = dates[split].strftime("%Y-%m-%d")
        # Candidate tuning uses TRAIN window only; no test P&L in selection.
        candidates: list[tuple[float, float, float]] = []
        for multiplier in cfg.stop_candidates:
            training_cfg = replace(cfg.base,
                                   atr_stop_multiple=multiplier,
                                   first_trade_date=training_trade_start)
            training = run_swing_backtest(train, training_cfg, corporate_actions=train_actions)
            net_return = training.final_equity_sek / cfg.base.initial_sek - 1.0
            # Penalize drawdown; don't optimize on test results.
            score = net_return - 1.5 * training.max_drawdown
            candidates.append((score, -multiplier, net_return))
        score, negative_multiplier, net_training_return = max(candidates)
        chosen = -negative_multiplier

        test_cfg = replace(cfg.base, atr_stop_multiple=chosen,
                           first_trade_date=test_trade_start)
        tested = run_swing_backtest(through_test, test_cfg, corporate_actions=test_actions)
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
                corporate_actions=test_actions
            )
            sensitivities[f"{cost_bps:g}bps"] = round(
                (stressed.final_equity_sek / cfg.base.initial_sek - 1.0) * 100, 6
            )
        fold_trades += len(tested.fills)
        folds.append(WalkForwardFold(
            number=len(folds) + 1,
            train_end=dates[split - 1].strftime("%Y-%m-%d"),
            test_start=test_trade_start,
            test_end=dates[split + cfg.test_sessions - 1].strftime("%Y-%m-%d"),
            chosen_atr_stop_multiple=chosen,
            training_net_return_pct=round(net_training_return * 100, 6),
            test_net_return_pct=round(net * 100, 6),
            test_drawdown_pct=round(tested.max_drawdown * 100, 6),
            test_hypothetical_fills=len(tested.fills),
            test_commissions_sek=round(tested.total_fees_sek, 4),
            cost_stress_returns_pct=sensitivities,
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
    )
