# Avanza StockBot: fixed-length rolling walk-forward protocol

**2026-10-30. BACKTEST ONLY; no inference of profitable live trading.**

Earlier implementation described `train_sessions=504` as a rolling window
but inadvertently used all available history from dataset inception on
later folds. That was **expanding-window** fitting, not the documented
rolling-window procedure.

This research branch now constructs each fold from exactly the configured
number of consecutive XSTO sessions.

- Fold `k` trains on sessions
  `[test_start - train_sessions, test_start - 1]`.
- Indicator warmup uses only the first 200 sessions **of that fold's train
  window**, not a stale date from dataset inception.
- The selected ATR stop multiplier is derived solely from that fold's
  training window; test results cannot influence the selection.
- Training corporate actions are filtered to ex-dates **inside the train
  window**. OOS testing sees only events occurring at or before its
  historical test end.
- Test trade decisions start at the test fold's first session after
  historical warmup; positions and initial cash are reset for each fold.
- The `WalkForwardFold` output records `train_start` and `train_end`
  so a reviewer can reproduce its boundaries. Test windows are disjoint
  with no training/test overlap *within each fold*. Earlier OOS periods
  may legitimately later become training observations.
- Synthetic regression tests reject expanding-window regressions and
  reject tuning on the same future OOS observations.

**Not completed:** dynamic IPO/delisting fold-aware backtesting, licensed
historical Swedish equities, non-survivor historical constituents, split
cash adjustments outside the supported action types, independently verified
GI benchmark history, true auction pricing, real paper-forward outcomes and
broker execution permissions. No automated funding or live trading.
