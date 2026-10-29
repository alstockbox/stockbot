# Avanza StockBot — corporate actions and halted-session research contract

**Status: IMPLEMENTERAT + lokalt testat, ej marknadsdataverifierat, ej live/paper driftsatt.**
2026-10-29. Scope: this StockBot equity swing branch ONLY, not MT5/forex/crypto.

## Data requirements
- OHLCV prices must be **RAW** (unadjusted) if `--actions-json` is used.
  Supplying split-adjusted OHLCV and applying a split event again would
  double-count the split. Reject adjusted-price manifests with actions.
- Every event is tied to a known instrument and a real market session.
  `announced_at` must precede 09:00 Europe/Stockholm on the ex-date.
- Events are only trusted as synthetic research inputs until provider terms,
  as-of metadata, corporate-action completeness and symbol histories have
  been independently checked. A JSON file is NOT evidence of data rights.
- Trading suspended or zero-volume session: no order is filled there; close
  marks may be stale and require separate market-data handling.

## Supported actions

**SPLIT**: Before the ex-date opening auction, multiply owned whole shares
by numerator / denominator, and divide outstanding stop levels by that
factor. The indicator history is made split-comparable while executions and
mark-to-market use RAW traded prices. Fractional resulting holdings fail
closed until an audited cash-in-lieu treatment is available.

**DIVIDEND**: Entitlement to gross dividend is captured from holdings
carried into the ex-date. It is booked as a receivable in equity, NOT as
spendable cash. It converts to cash on the first modeled session on/after
`pay_date`, even if those shares were later sold. Outstanding receivables
appear separately in results.

**Unsupported**: rights issues, stock mergers, spin-offs, takeovers, special
distributions, variable taxes, fractional share settlement, exchange halts
without a proper mark, changes in symbol listings and forced delistings.
Such events must not be silently ignored in research-grade datasets.

## Example input (synthetic template only)

`data/actions.json`:
```json
[
  {
    "symbol": "VOLV.B",
    "kind": "SPLIT",
    "ex_date": "2025-06-10",
    "announced_at": "2025-06-01T08:00:00Z",
    "numerator": 2,
    "denominator": 1
  },
  {
    "symbol": "VOLV.B",
    "kind": "DIVIDEND",
    "ex_date": "2025-09-10",
    "announced_at": "2025-09-01T08:00:00Z",
    "dividend_per_share_sek": 5.0,
    "pay_date": "2025-09-16"
  }
]
```

Dates and amounts above are **made-up examples**, not actual Volvo
corporate-action claims or a ready-made valid dataset.

```sh
python scripts/run_swing_research.py \
  --csv data/stockholm_raw_ohlcv.csv \
  --manifest data/provenance.json \
  --actions-json data/actions.json \
  --capital-sek 1000 \
  --output reports/avanza_swing.json
```

Add `--walkforward` to use fold-isolated OOS research; the same events
are passed to train/test windows only when ex-date has actually occurred.
No source authorization, out-of-sample alpha or broker ability is implied.

## Verification and next blockers

- Deterministic synthetic regression tests validate 2:1 splits, dividends
  settled and outstanding, unsupported fractional conversions, unknown
  and late-announced events, zero-volume halts, JSON import and test-window
  event scoping.
- No current real Nasdaq Stockholm dataset has been accepted or tested.
- Historical constituent/delist universe, actual ex/pay dates, tax/withholding,
  auction/slippage/partial fills, provider licenses and benchmark remain P0.
- Max 1,000 SEK live pilot only after separate explicit user approval,
  positive forward paper evidence and permitted broker/API channel.
