# Avanza StockBot: conservative daily-liquidity research v1

**Status 2026-10-29: implemented in isolated StockBot research branch.**
This is not order routing or evidence of executable fills.

## Why this exists
The old backtest could instantly buy/sell arbitrarily many shares during
a low-volume session if reported volume was merely nonzero. That produces
unrealistically optimistic returns, especially for Swedish small caps.

## Guards in `src/stockbot/swing/engine.py`
- For a new entry, use the **previous session's** known 20-session median
  daily SEK turnover and 20-session median share volume. This avoids using
  today's not-yet-observable turnover to choose a signal.
- Default required historic median turnover: **SEK 100,000/day** (tentative
  research filter, NOT a market-liquidity guarantee).
- Default maximum participation: **2%** of a day's reported share volume,
  shared by all fills for that instrument on that day.
- Future day's full-session volume is used to cap *retrospective simulated
  fills*, never to select signals. Because full-session volume is not
  observable at the opening auction, this cap is a conservative
  retrospective approximation, **not** proof that the opening or stop
  price would have been executable in real trading.
- Existing position exits execute first. Insufficient volume causes
  a partial SELL, with the rest carried as `exit_pending` into later
  sessions. Follow-up SELLs use next session's OPEN and available volume.
  Minimum commission applies to each separate partial fill.
- Held shares that cannot exit before dataset end are reported in
  `unresolved_exit_shares`; no invented forced liquidation.
- Zero-volume days generate no BUY/SELL fills; their unconfirmed raw OHLC
  is not used to create phantom equity or adjust an open position's stop.
- No same-day reentry after a completed exit. Buying and selling the same
  shares cannot exceed the daily participation cap.
- No withdrawals, margin, shorts, live order adapter or deposits.

## Reproducible offline CLI

```sh
python scripts/run_swing_research.py --csv data/xsto_raw.csv \
  --manifest data/provenance.json --capital-sek 1000 \
  --min-prior-turnover-sek 100000 --max-daily-volume-fraction 0.02 \
  --output reports/xsto-liquidity.json
```

Parameters are configurable for explicit cost sensitivity, but must not
be optimized on the same untouched OOS fold used to judge alpha.
`--walkforward` uses the same fixed liquidity configuration on all folds.

## Remaining realism blockers
- Full-day OHLCV does NOT reveal available order-book depth, spread,
  auction matching volume, queue priority, minute-level halt times, or
  true stop-order execution. A 2% daily volume cap cannot guarantee fills.
- Broker contract, Avanza's API permission, market data rights and 2026
  commission schedule need independent validation.
- A zero-volume bar can still be misleading corporate-action data.
  A suspension may be accompanied by stale quotes or indicative values.
- Price-time priority, tick size, lot sizes, partial-order cancellation,
  rejected orders, exchange auctions, tax and intraday spread are not modeled.
- No Swedish live market data, licensed historical constituent universe,
  delistings or independently validated OMX total-return benchmark yet.
- All reports remain BACKTEST_ONLY/UNVERIFIED. Forward paper trading and
  the separately approved SEK 1,000 live pilot have NOT begun.

## Acceptance test coverage
- Reject illiquid entry using only prior-close data.
- Day-share volume cap for hypothetical fills.
- Partial stop exits with next-day retry.
- Unresolved exit shares at dataset end.
- Halt does not reset exit intent or mark phantom prices.
- Invalid liquidity configuration rejected.
