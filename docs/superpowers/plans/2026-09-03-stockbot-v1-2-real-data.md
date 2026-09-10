# StockBot V1.2 Real Market Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect StockBot to real EOD market data through Tiingo and a zero-key bootstrap adapter, persist immutable reproducible snapshots, and feed those snapshots into the existing V1 Alpha Arena without weakening data-grade promotion gates.

**Architecture:** Provider adapters normalize external responses into one canonical EOD schema. A download service validates complete universes and persists immutable CSV+JSON snapshots whose grade/provenance flows unchanged into the existing V1 training pipeline. Network transports are injectable so tests and CI remain offline and deterministic.

**Tech Stack:** Python 3.11+, standard-library urllib/json/pathlib/hashlib, pandas, existing StockBot numpy/scikit-learn stack, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-03-stockbot-v1-2-real-data-design.md`

## Global Constraints

- V1.2 remains research/backtest/shadow/paper-only.
- Tiingo token comes from configuration/environment and is never serialized or logged.
- Yahoo bootstrap data is always BOOTSTRAP and never promotable.
- Raw and adjusted price fields are preserved separately.
- Partial universe downloads fail closed.
- Snapshot directories are immutable.
- Existing V0/V1 APIs and tests must remain green.
- CI must require no external API token or network request to market-data providers.

---

### Task 1: Extend data grades and canonical market schema

**Files:**
- Modify: `src/stockbot/data/schemas.py`
- Create: `src/stockbot/data/market_schema.py`
- Test: `tests/data/test_market_schema.py`

**Interfaces:**
- Produces `DataGrade.DEMO`, `DataGrade.BOOTSTRAP`, `DataGrade.RESEARCH`.
- Produces `CANONICAL_BAR_COLUMNS` and `validate_canonical_bars(frame)`.

- [ ] Write tests requiring BOOTSTRAP grade and raw/adjusted columns to coexist.
- [ ] Run focused tests and confirm RED because schema additions do not exist.
- [ ] Implement the enum extension and canonical validation for timestamp/symbol uniqueness, finite positive OHLC, non-negative volume and valid date ordering.
- [ ] Run focused tests and full suite.
- [ ] Commit `feat: add canonical real market data schema`.

### Task 2: Tiingo provider

**Files:**
- Create: `src/stockbot/data/providers/http.py`
- Create: `src/stockbot/data/providers/tiingo.py`
- Test: `tests/data/test_tiingo_provider.py`

**Interfaces:**
- `HttpTransport.get_json(url, headers) -> object`.
- `TiingoProvider(token, transport=None).fetch_bars(symbol, start, end) -> pd.DataFrame`.

- [ ] Write tests for missing token, Authorization header, start/end query parameters and normalization of raw/adjusted/dividend/split fields.
- [ ] Verify RED.
- [ ] Implement urllib-based transport and Tiingo adapter with injected transport support.
- [ ] Verify malformed/empty provider responses raise explicit `ProviderError`.
- [ ] Run full suite and commit `feat: add Tiingo EOD provider`.

### Task 3: Yahoo bootstrap provider

**Files:**
- Create: `src/stockbot/data/providers/yahoo_bootstrap.py`
- Test: `tests/data/test_yahoo_bootstrap_provider.py`

**Interfaces:**
- `YahooBootstrapProvider(transport=None).fetch_bars(symbol, start, end) -> pd.DataFrame`.
- Provider exposes immutable `grade == DataGrade.BOOTSTRAP`.

- [ ] Write fixture-based tests for chart timestamps, quote arrays, adjusted close, dividends and splits.
- [ ] Verify RED.
- [ ] Implement chart request construction with `period1`, `period2`, daily interval and corporate-action events.
- [ ] Reject missing timestamps/results and preserve BOOTSTRAP grade.
- [ ] Run full suite and commit `feat: add bootstrap market data provider`.

### Task 4: Immutable snapshot store and manifests

**Files:**
- Create: `src/stockbot/data/snapshots.py`
- Test: `tests/data/test_snapshots.py`

**Interfaces:**
- `SnapshotManifest` dataclass.
- `SnapshotStore(root).write(bars, manifest_input) -> MarketSnapshot`.
- `SnapshotStore(root).load(snapshot_id) -> MarketSnapshot`.

- [ ] Test deterministic content fingerprint, sorted symbols, row/date summaries and absence of secret fields.
- [ ] Test writing a duplicate snapshot ID refuses overwrite.
- [ ] Verify RED.
- [ ] Implement CSV+JSON snapshot directories with UTC timestamp + short fingerprint IDs.
- [ ] Round-trip load and canonical validation.
- [ ] Run full suite and commit `feat: add immutable market snapshot store`.

### Task 5: Download orchestration and complete-universe guarantees

**Files:**
- Create: `src/stockbot/data/download.py`
- Test: `tests/data/test_download.py`

**Interfaces:**
- `download_market_snapshot(provider, symbols, start, end, store, grade=None, provenance=None) -> MarketSnapshot`.

- [ ] Test symbol deduplication/sorting, invalid ranges and empty universe rejection.
- [ ] Test one failed/missing symbol aborts the entire snapshot.
- [ ] Verify RED.
- [ ] Implement fetch/concat/validate/snapshot orchestration with explicit provider grade rules.
- [ ] Ensure Yahoo cannot be upgraded from BOOTSTRAP and Tiingo defaults to BOOTSTRAP unless explicit research provenance is supplied.
- [ ] Run full suite and commit `feat: orchestrate reproducible market downloads`.

### Task 6: Snapshot-to-training integration

**Files:**
- Create: `src/stockbot/research/market_training.py`
- Test: `tests/integration/test_market_training.py`

**Interfaces:**
- `train_snapshot(snapshot, model_configs=None, horizon=5) -> TrainingRun`.

- [ ] Test canonical snapshot bars feed existing `run_training_research` without duplicate ML logic.
- [ ] Test BOOTSTRAP snapshot produces no champion candidate even when an experiment score ranks first.
- [ ] Verify RED.
- [ ] Map snapshot manifest to existing `DatasetMetadata` and invoke V1 training pipeline.
- [ ] Run full suite and commit `feat: train Alpha Arena from market snapshots`.

### Task 7: CLI, configuration and documentation

**Files:**
- Create: `scripts/run_market_training.py`
- Create: `.env.example`
- Modify: `README.md`
- Test: `tests/integration/test_market_cli.py`

**Interfaces:**
- CLI arguments: `--provider`, `--symbols`, `--start`, `--end`, `--snapshot-root`, `--train`.
- Tiingo reads `TIINGO_API_TOKEN` from environment.

- [ ] Test CLI parser and missing Tiingo token without network.
- [ ] Verify RED.
- [ ] Implement provider selection, snapshot summary and optional training report.
- [ ] Document Tiingo setup, bootstrap caveats and example commands.
- [ ] Run full suite and both existing demos.
- [ ] Commit `feat: add real market data training CLI`.

### Task 8: CI and final verification

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] Add an offline V1.2 fixture/snapshot smoke test to CI; never inject secrets.
- [ ] Run `python -m compileall -q src scripts`.
- [ ] Run `pytest -q` with zero failures.
- [ ] Run `python scripts/run_demo.py`.
- [ ] Run `python scripts/run_training_demo.py`.
- [ ] Scan repository for token assignments, Authorization values and live broker execution code.
- [ ] Compare feature branch with `main` and verify only V1.2-related files changed.
- [ ] Commit `ci: verify real market data pipeline`.