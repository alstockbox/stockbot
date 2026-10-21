# StockBot V1.2 Real Market Data Bootstrap Design

## Mission

V1.2 connects StockBot's V1 training engine to real historical market data while preserving the distinction between bootstrap-quality data and research-grade point-in-time data. The goal is to make `download -> validate -> snapshot -> train -> Alpha Arena -> report` reproducible and safe enough to begin large-scale historical training without granting bootstrap data authority to promote a production champion.

V1.2 remains research/backtest/shadow/paper-only. It introduces no live-money execution.

## Provider strategy

StockBot uses a provider-neutral interface. Two concrete adapters are introduced:

1. `TiingoProvider` — the preferred serious EOD provider. It consumes Tiingo's documented daily endpoint and normalizes raw and adjusted OHLCV plus dividends and splits. The token is read only from configuration/environment and is never persisted in repository files or snapshot manifests.
2. `YahooBootstrapProvider` — a zero-key bootstrap adapter around Yahoo's public chart endpoint. Because that endpoint is unofficial and provides no survivorship/point-in-time guarantee, all output is marked `BOOTSTRAP` and can never satisfy champion-promotion requirements.

Provider errors must be explicit and typed. Empty responses, malformed records, HTTP errors and missing credentials must never silently become empty training datasets.

## Data grades

V1.2 extends data quality semantics to:

- `DEMO`: synthetic/local data used only for tests and examples.
- `BOOTSTRAP`: real market prices from a source without research-grade survivorship/point-in-time guarantees.
- `RESEARCH`: source/data contract considered suitable for serious historical experiments when the caller explicitly supplies research-grade metadata.

Only `RESEARCH` datasets can be eligible for champion promotion. `BOOTSTRAP` and `DEMO` data may be ranked in the Arena but cannot produce a promotion candidate.

## Canonical EOD schema

Each normalized bar contains at least:

- `timestamp`
- `symbol`
- `open`, `high`, `low`, `close`, `volume`
- `adj_open`, `adj_high`, `adj_low`, `adj_close`, `adj_volume` when available
- `div_cash`
- `split_factor`
- `provider`
- `retrieved_at`

Raw and adjusted values remain separate. The feature/training layer may choose adjusted values explicitly; provider normalization must never overwrite raw prices.

## Tiingo adapter

`TiingoProvider` uses:

`GET https://api.tiingo.com/tiingo/daily/<ticker>/prices?startDate=<YYYY-MM-DD>&endDate=<YYYY-MM-DD>`

Authentication is supplied through an `Authorization: Token <token>` header. The adapter accepts an injectable transport for deterministic tests and defaults to a standard-library HTTPS transport in production.

Tiingo fields normalized:

- `date` -> `timestamp`
- `open/high/low/close/volume`
- `adjOpen/adjHigh/adjLow/adjClose/adjVolume`
- `divCash`
- `splitFactor`

The provider grade defaults to `BOOTSTRAP` for EOD-only ingestion. A caller may only assign `RESEARCH` when it supplies an explicit research-grade universe/point-in-time provenance object; the adapter itself does not make that claim.

## Yahoo bootstrap adapter

`YahooBootstrapProvider` uses the chart endpoint with explicit Unix `period1` and `period2`, daily interval and dividend/split events. It extracts quote OHLCV, adjusted close and corporate-action events when present.

The adapter is deliberately isolated and marked unofficial. Its dataset metadata is always `BOOTSTRAP`. It cannot be configured to return `RESEARCH` grade.

## Universe snapshots

Every download is tied to an explicit symbol universe. The snapshot manifest stores:

- sorted symbol list
- requested start/end dates
- provider name
- data grade
- retrieval timestamp
- row count
- first/last observation per symbol
- dataset fingerprint
- schema version
- optional notes/provenance

The universe list is immutable inside the manifest. This makes survivorship assumptions visible and reproducible instead of implicitly using today's symbols.

## Snapshot store

`SnapshotStore` writes one immutable snapshot directory per dataset under a user-selected root, for example:

```text
snapshots/
  20260903T001500Z-tiingo-a1b2c3d4/
    bars.csv
    manifest.json
```

CSV is the required baseline because StockBot currently has no Parquet dependency. A future optional Parquet backend can be added without changing provider/training interfaces.

Snapshot IDs include UTC retrieval time and a short content fingerprint. Existing snapshot directories are never overwritten.

Manifest fingerprints are deterministic over canonical bar content + universe + provider + grade + date range + schema version. API tokens and secrets are excluded from all fingerprints and manifests.

## Download orchestration

`download_market_snapshot(provider, symbols, start, end, store, metadata)` performs:

1. validate the symbol universe and date range;
2. fetch each symbol through the provider;
3. normalize into the canonical schema;
4. validate OHLC consistency, finite prices, non-negative volume, unique `(timestamp, symbol)` rows and requested bounds;
5. sort deterministically;
6. create immutable manifest/fingerprint;
7. persist the snapshot;
8. return a `MarketSnapshot` object.

Partial downloads fail closed by default. V1.2 does not silently continue when one requested symbol fails, because that changes the universe and can bias results.

## Training integration

`train_snapshot(snapshot, ...)` loads canonical bars and passes them into the existing V1 `run_training_research` pipeline. The snapshot's data grade becomes the `DatasetMetadata.grade` used by Arena promotion gates.

The V1.2 CLI can therefore run:

```text
provider -> download -> snapshot -> existing V1 panel/features/labels -> purged walk-forward model zoo -> Alpha Arena -> report
```

No duplicate model-training implementation is introduced.

## CLI

A new script `scripts/run_market_training.py` supports:

- provider selection: `tiingo` or `yahoo-bootstrap`
- comma-separated symbols
- start/end dates
- snapshot directory
- optional `--train`

For Tiingo, `TIINGO_API_TOKEN` is read from the environment. Missing token exits with a concise error before any request is sent.

CLI output includes snapshot ID, provider, grade, symbols, rows, fingerprint and—when training is requested—model leaderboard/champion eligibility.

## Security and secrets

- API tokens are accepted only via environment/config objects.
- Tokens are never logged, serialized to manifests, included in exception strings or committed to GitHub.
- `.env.example` may document the variable name only, never a value.
- HTTP query strings do not contain the Tiingo token; authorization uses headers.

## Testing

TDD coverage includes:

- Tiingo URL/header construction and response normalization;
- Yahoo bootstrap parsing and forced BOOTSTRAP grade;
- missing-token rejection;
- raw vs adjusted values preserved separately;
- split/dividend normalization;
- deterministic snapshot fingerprints;
- immutable snapshot directories;
- manifest secret exclusion;
- duplicate/invalid bar rejection;
- all-requested-symbols requirement;
- snapshot-to-training integration;
- bootstrap data blocked from champion promotion;
- existing V0/V1 test suite remains green;
- GitHub Actions runs tests and offline demos without requiring any API secret.

## Non-goals

V1.2 does not yet add:

- Tiingo fundamentals ingestion;
- news/LLM market-event ingestion;
- automatic daily scheduling;
- massive universe discovery from Tiingo supported-tickers files;
- survivorship-bias-free historical constituent reconstruction;
- live broker execution.

Those become subsequent milestones after the real-price ingestion path is stable.

## Success criteria

V1.2 is complete when StockBot can fetch real EOD data through Tiingo when a token is configured, fetch bootstrap real data without a token through the bootstrap adapter, persist a deterministic immutable snapshot, reload that snapshot, run it through the existing V1 Alpha Arena, and provably prevent non-RESEARCH data from producing a promotable champion.