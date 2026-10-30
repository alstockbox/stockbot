# Avanza StockBot — lifecycle replay for IPO and cash-only delisting

**2026-10-30. RESEARCH ONLY.** The Avanza equity swing project remains
separate from the MT5/Vantage forex/crypto bot. No live order endpoint,
account login, paid data or money transfers are used.

## What was changed

The existing `run_swing_backtest` now supports an **opt-in** lifecycle
execution path alongside the unchanged fixed-panel path.

- A supplied point-in-time `--universe-json` with ISINs, first listing
  sessions, last listed sessions, disclosure timestamps and ACTIVE/DELISTED
  status is audited *before* any trade simulation.
- Stocks may enter the market after the backtest's first session, or
  disappear after a documented delisting. No forward or backward padding,
  fabricated pre-IPO prices or imagined post-delist closes.
- New entries require a valid listing on **both the previous** and current
  XSTO trading sessions, with 200 genuine symbol-specific indicator bars.
- No new purchases on an explicitly known final tradable day.
- The market-data union must contain **every** XSTO trading session from
  start to finish. All present symbols must contain valid, contiguous
  observations throughout their individual listed periods.
- Existing partial-exit, cash, stop, commission, liquidity, split and
  dividend controls remain enabled; historic symbol dates are checked.
- Each DELISTED stock ending inside the sample must have an explicit
  `--settlements-json` entry, **even when the portfolio holds none**.
  Missing, inconsistent or noncash settlement evidence rejects the run.
- A `CASH_ONLY` delisting changes outstanding holdings into a cash
  *receivable* **after final market close**, and into spendable cash only
  on/after the declared payment day. A zero-cash delisting removes the
  shares at zero recovery, not at the last quoted price. This is an
  accounting event, **not** a `SELL` fill.
- The source/reference and final cash consideration must have been
  established **before the final session opens** to avoid using
  later information to decide whether to buy on that date.

## JSON event format (invented illustration, not a real Swedish company)

```json
[
  {
    "symbol": "TEST",
    "last_session": "2025-12-19",
    "cash_per_share_sek": 2.0,
    "pay_date": "2026-01-15",
    "known_at_utc": "2025-12-18T07:00:00Z",
    "source_reference": "independently checked cash offer document",
    "currency": "SEK",
    "settlement_kind": "CASH_ONLY"
  }
]
```

`cash_per_share_sek=0` requires actual evidence that equity recovery is
zero. An assumed zero recovery without supporting evidence is not
adequate for historical research.

Offline synthetic or independently authorized data:

```sh
python scripts/run_swing_research.py \
  --csv data/xsto_raw_bars.csv \
  --manifest data/xsto_provenance.json \
  --universe-json data/point_in_time_universe.json \
  --settlements-json data/delist_cash_settlements.json \
  --capital-sek 1000 --output reports/lifecycle_research.json
```

The report carries `lifecycle_events`, `delisted_shares`,
`delist_cash_paid_sek`, `delist_receivable_sek` and a separate
structural `universe_audit` receipt. All reports retain
`BACKTEST_ONLY`, `orders_sent=0`, and
`license_independently_verified=false`.

## Deliberately blocked / incomplete

- **Dynamic-universe walk-forward is not yet supported.** Using
  `--walkforward` with IPO/delisting transitions fails closed.
  It requires fold-scoped, nonleaking universe provenance and settlement
  handling before it can be accepted for OOS testing.
- Mergers involving shares, spin-offs, renaming/ISIN reuse, rights issues,
  multiple share classes, compulsory redemptions, unrecorded bankruptcies,
  tax withholding, unverifiable payouts and record-date uncertainty.
- True intraday auction depth/spread, order queuing, real broker fees,
  share fractional settlements and execution reconciliation.
- A user-authored universe file **cannot independently prove**
  survivorship-free selection. A vendor dataset that omits bankrupt names
  cannot be detected from its own incomplete list. Separate trusted
  historical reference data and licensing are P0.

No paper or live order routing is connected; no automatically approved
funding. A **max SEK 1,000** live pilot requires separately documented
forward-paper evidence and the user's explicit permission.
