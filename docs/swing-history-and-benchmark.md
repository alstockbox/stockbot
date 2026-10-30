# StockBot Avanza — point-in-time universe and GI-index benchmark, research only

**Datum:** 2026-10-30. **Status:** Code/fixture validation only; no live or paper orders.

## Why this gate is necessary

A list of companies that exist **today** is an invalid historical investment
universe. A good-looking backtest can reflect *survivorship bias*: disappeared,
bankrupt or delisted companies are excluded without explanation. StockBot must
NOT claim a validated excess return from such a selection.

### Auditing historical listings and delistings

`src/stockbot/swing/universe.py` and
`scripts/audit_swing_universe.py` accept two separately supplied,
timestamped, licensed/source-referenced inputs:
- Daily XSTO OHLCV file: `date,symbol,open,high,low,close,volume`.
- Universe JSON: each symbol, ISIN, first tradable date, last tradable
  date if delisted, announcement timestamps, and ACTIVE/DELISTED status.
- Data provenance JSON: provider, terms reference, UTC collection stamp,
  explicit SEK currency and RAW/SPLIT_ADJUSTED price basis.

Example **SYNTHETIC structure only** (the ISIN below is invented and is
not an actual investable instrument):
```json
[
  {
    "symbol": "TEST",
    "isin": "SE0000000001",
    "first_session": "2025-01-02",
    "last_session": null,
    "first_known_at_utc": "2024-12-01T00:00:00Z",
    "last_known_at_utc": null,
    "status": "ACTIVE"
  }
]
```

Run:
```sh
python scripts/audit_swing_universe.py --csv data/raw.csv \
  --universe-json data/universe.json \
  --manifest data/provenance.json --output reports/universe_audit.json
```

Checks: no phantom prices before IPO/after delisting, no missing daily bars
within the supported listing lifespan, unique symbol/ISIN mappings,
disclosures not dated in the future relative to the dataset, XSTO sessions,
valid OHLCV and deterministic SHA-256 source/manifest fingerprints.

**Crucial limitation:** This only checks *internal structural consistency*.
A dataset that intentionally omits every ruined/delisted company cannot be
independently detected using its own self-reported universe list.
`historical_universe_independently_verified=false` always. Independent
time-stamped records from the data vendor are mandatory.

**Do not bypass the simulator limitation:** The current swing engine
requires symbols to appear in every trading session. A valid historical
IPO/delisting audit may therefore pass while
`compatible_with_current_fixed_panel_engine=false`.
`run_swing_research.py --universe-json` refuses such a replay.
No synthetic fill, padded price or forced liquidation is fabricated.
A proper event-driven lifecycle/settlement engine is P0 next.

### Independent OMX index comparison

`src/stockbot/swing/benchmark.py` accepts **licensed, user-supplied**
daily `date,close` index values and the matching benchmark manifest. It
compares each held-out OOS fold with a SEK **gross total-return** index
anchored at the prior train-end close, not the first test close.
Reject missing XSTO days, unknown codes, wrong currency, price-only indexes,
duplicates, stale collection timestamps or nonpositive prices.
Identities considered: OMXSBGI, OMXS30GI, OMXSGI. Choose index by
actual stock-universe exposure, not whichever gives the best backtest result.

Run (data paths are placeholders; do not download/redistribute without rights):
```sh
python scripts/run_swing_research.py --csv data/raw.csv \
  --manifest data/provenance.json \
  --universe-json data/universe.json \
  --walkforward --benchmark-csv data/index_gi.csv \
  --benchmark-manifest data/index_rights.json \
  --output reports/compare_oos.json
```

All benchmark results are tagged:
`mode=UNVERIFIED_OOS_BENCHMARK_COMPARISON`,
`license_independently_verified=false`,
`external_index_history_verified=false`,
`alpha_claim_permitted=false`.
They are **arithmetic comparisons, not profit proofs**.

Benchmark manifest requires the exact six fields:
```json
{
  "index_code": "OMXSBGI",
  "source": "licensed data provider",
  "usage_rights_reference": "independently checked usage terms",
  "collected_at_utc": "2026-10-29T22:00:00Z",
  "currency": "SEK",
  "return_basis": "GROSS_TOTAL_RETURN"
}
```

### Nasdaq source checks (not data access authorization)

- Nasdaq Nordic HistoricalView:
  https://www.nasdaq.com/products/data/nordic-baltic/historicalview
- Nasdaq official index description OMXSBGI:
  https://indexes.nasdaqomx.com/index/overview/omxsbgi
- Nasdaq market-data product catalog:
  https://www.nasdaq.com/products/data/market-data-catalog

**Do not automatically scrape Nasdaq or Yahoo** to satisfy this dependency.
Costs, coverage, licenses, user account permissions and allowed automated
uses must be verified before accepting a real dataset. No monthly service
charges may be committed without the user's approval.

### Remaining P0 gates
1. Provider/rights verification for RAW bars, histories, corporate actions,
   delisted instruments and suitable GI benchmark; no invented free API.
2. True lifecycle-aware event-driven replay including actual delisting
   distributions, suspension/cash settlements, ticker history and IPO dates.
3. Real-world order spread, auction execution and applicable Avanza fees.
4. Held-out historical OOS testing, then prolonged timestamped paper
   trading with no broker order placement.
5. MAX SEK 1,000 live pilot **only** after independent approval. No auto
   funding; Forex/MT5 project kept separate.
