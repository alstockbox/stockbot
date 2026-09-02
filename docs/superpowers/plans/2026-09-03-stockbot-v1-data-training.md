# StockBot V1 Data & Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a provider-neutral, point-in-time-aware multi-symbol training system that can generate causal panel features/labels, run purged walk-forward model experiments, and rank only out-of-sample challengers in StockBot Arena.

**Architecture:** V1 extends the existing V0 package without changing its execution safety boundary. Canonical market data is normalized into a MultiIndex panel, features and labels are produced causally, model families implement one common interface, and an experiment runner assembles strictly time-ordered OOS predictions before any strategy metrics or promotion decision is calculated.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, pydantic, scipy, pytest, hashlib/json from stdlib. No new paid-data dependency and no GPU dependency.

**Spec:** `docs/superpowers/specs/2026-09-03-stockbot-v1-data-training-design.md`

## Global Constraints

- V1 remains research/backtest/shadow/paper-only; no live-money execution path.
- All trading performance claims must use time-ordered OOS predictions.
- Point-in-time records with `available_time` later than the decision timestamp must be excluded.
- Demo/non-research-grade datasets may run experiments but cannot qualify a champion for live-capital consideration.
- No random train/test split is permitted for trading-performance evaluation.
- Model randomness must be deterministic from an explicit seed.
- Dataset and experiment metadata must be fingerprinted and reproducible.
- Existing V0 tests must remain green.

---

### Task 1: Canonical data schemas and validation

**Files:**
- Create: `src/stockbot/data/__init__.py`
- Create: `src/stockbot/data/schemas.py`
- Create: `src/stockbot/data/validation.py`
- Test: `tests/data/test_schemas.py`
- Test: `tests/data/test_validation.py`

**Interfaces:**
- Produces: `DataGrade`, `DatasetMetadata`, `validate_bars(frame)`, `as_of_filter(frame, decision_time)`.

- [ ] Write a failing test that accepts canonical columns `symbol, timestamp, open, high, low, close, volume` and rejects duplicate `(symbol,timestamp)` rows, nonpositive close, negative volume, or unsorted symbol/timestamps.
- [ ] Write a failing test proving a record with `available_time > decision_time` is excluded by `as_of_filter`.
- [ ] Run focused tests and verify RED because `stockbot.data` is missing.
- [ ] Implement immutable metadata/data-grade types and strict validation with no backward filling or silent duplicate removal.
- [ ] Run focused tests and full suite; require zero failures.
- [ ] Commit `feat: add canonical point in time data contracts`.

### Task 2: Provider-neutral local adapter and panel builder

**Files:**
- Create: `src/stockbot/data/providers/__init__.py`
- Create: `src/stockbot/data/providers/base.py`
- Create: `src/stockbot/data/providers/local.py`
- Create: `src/stockbot/data/panel.py`
- Test: `tests/data/test_providers.py`
- Test: `tests/data/test_panel.py`

**Interfaces:**
- Produces: `MarketDataProvider.load_bars(symbols,start,end)`, `LocalFrameProvider`, `build_panel(frame) -> pd.DataFrame` indexed by `(timestamp,symbol)`.

- [ ] Write a failing provider contract test that preserves symbols/date bounds and dataset grade.
- [ ] Write a failing panel test proving input row order does not change canonical output and duplicate observations are rejected.
- [ ] Verify RED.
- [ ] Implement provider protocol, in-memory/local provider, and canonical sorted MultiIndex panel builder.
- [ ] Verify GREEN and full suite.
- [ ] Commit `feat: add provider neutral panel data layer`.

### Task 3: Cross-sectional features and multi-horizon labels

**Files:**
- Create: `src/stockbot/features/cross_sectional.py`
- Create: `src/stockbot/ml/labels.py`
- Test: `tests/features/test_cross_sectional.py`
- Test: `tests/ml/test_labels.py`

**Interfaces:**
- Produces: `add_cross_sectional_features(panel) -> pd.DataFrame`.
- Produces: `make_panel_labels(panel, horizons=(1,5,20), benchmark=None) -> pd.DataFrame`.

- [ ] Write failing tests for same-date percentile ranks of momentum, volatility and volume that do not use future dates.
- [ ] Write failing tests for 1/5/20-day forward return and adverse-excursion labels, with tail rows dropped only when the horizon is unavailable.
- [ ] Verify RED.
- [ ] Implement grouped-by-symbol historical transforms and grouped-by-date cross-sectional ranking; implement labels strictly with forward shifts inside each symbol.
- [ ] Verify GREEN and full suite.
- [ ] Commit `feat: add cross sectional features and labels`.

### Task 4: Purged walk-forward validation

**Files:**
- Create: `src/stockbot/ml/purged_cv.py`
- Test: `tests/ml/test_purged_cv.py`

**Interfaces:**
- Produces: `PurgedWalkForwardSplitter(train_periods,test_periods,label_horizon,embargo_periods).split(index)` yielding ordered integer train/test arrays.

- [ ] Write a failing test proving every training observation precedes test observations and all training rows whose label horizon overlaps the test window are purged.
- [ ] Write a failing test proving embargo periods are excluded after each test fold before they may enter a later training fold.
- [ ] Verify RED.
- [ ] Implement deterministic date-based splitting that works with repeated timestamps in a multi-symbol panel.
- [ ] Verify GREEN and full suite.
- [ ] Commit `feat: add purged walk forward validation`.

### Task 5: Expanded deterministic model zoo

**Files:**
- Modify: `src/stockbot/ml/challenger.py`
- Create: `src/stockbot/ml/models.py`
- Test: `tests/ml/test_model_zoo.py`

**Interfaces:**
- Produces: `ModelConfig(name, params, seed)`, `build_model(config)`, supported names `ridge`, `elastic_net`, `extra_trees`, `random_forest`, `hist_gb`.

- [ ] Write failing parameterized tests that every supported model fits and returns finite deterministic predictions for the same seed.
- [ ] Write a failing test that unknown models are rejected.
- [ ] Verify RED.
- [ ] Implement model factory using existing scikit-learn dependencies and keep preprocessing inside fold-local pipelines where required.
- [ ] Verify GREEN and full suite.
- [ ] Commit `feat: expand deterministic machine learning model zoo`.

### Task 6: OOS trainer and experiment artifacts

**Files:**
- Create: `src/stockbot/ml/artifacts.py`
- Create: `src/stockbot/ml/trainer.py`
- Test: `tests/ml/test_trainer.py`
- Test: `tests/ml/test_artifacts.py`

**Interfaces:**
- Produces: `dataset_fingerprint(frame, metadata) -> str`.
- Produces: `ExperimentArtifact` with dataset fingerprint, feature names, label name, model config, seed, fold count, OOS coverage and metrics.
- Produces: `train_oos(panel_features, labels, splitter, model_config) -> OOSResult`.

- [ ] Write failing tests proving predictions exist only on test rows and never on training-only rows.
- [ ] Write a failing deterministic fingerprint test where changing one value or grade changes the fingerprint.
- [ ] Write a failing test proving feature columns are fit/used within each training fold and prediction indexes match source rows.
- [ ] Verify RED.
- [ ] Implement fold-local training/prediction assembly and stable SHA-256 metadata fingerprints.
- [ ] Verify GREEN and full suite.
- [ ] Commit `feat: add reproducible out of sample training artifacts`.

### Task 7: Alpha Arena experiments and leaderboard

**Files:**
- Create: `src/stockbot/arena/experiments.py`
- Create: `src/stockbot/arena/leaderboard.py`
- Test: `tests/arena/test_experiments.py`
- Test: `tests/arena/test_leaderboard_v1.py`

**Interfaces:**
- Produces: `ExperimentConfig`, `run_model_experiment(...)`, `rank_experiments(results)`, `eligible_for_promotion(result, metadata)`.

- [ ] Write a failing test that compares at least three model configs on identical folds and sorts by OOS research score, not in-sample fit.
- [ ] Write a failing test where the highest raw return loses because of materially worse drawdown/turnover/stability.
- [ ] Write a failing test proving `DataGrade.DEMO` is never promotion-eligible.
- [ ] Verify RED.
- [ ] Implement OOS signal conversion, cost-aware backtesting through the existing V0 engine, robustness scoring, and promotion eligibility rules.
- [ ] Verify GREEN and full suite.
- [ ] Commit `feat: add multi model alpha arena`.

### Task 8: End-to-end V1 training pipeline and demo

**Files:**
- Create: `src/stockbot/research/training_pipeline.py`
- Create: `scripts/run_training_demo.py`
- Test: `tests/integration/test_training_pipeline.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `run_training_research(bars, metadata, model_configs=None, horizon=5) -> TrainingRun` containing dataset fingerprint, leaderboard, OOS coverage, champion candidate and data grade.

- [ ] Write a failing integration test using deterministic multi-symbol synthetic bars covering several years and at least five symbols.
- [ ] Assert all models in the leaderboard have positive OOS coverage, finite metrics and one shared dataset fingerprint.
- [ ] Assert demo-grade input returns no promotion-eligible champion even when one model ranks first.
- [ ] Verify RED.
- [ ] Implement orchestration from validation -> panel -> features -> labels -> purged folds -> model zoo -> OOS backtests -> leaderboard.
- [ ] Add CLI output that clearly labels synthetic/demo results as non-research-grade.
- [ ] Verify GREEN, run full suite and demo.
- [ ] Commit `feat: deliver StockBot V1 training pipeline`.

### Task 9: Final verification and branch handoff

**Files:**
- Modify only if verification finds defects.

- [ ] Run `pytest -q` and require zero failures across V0 + V1.
- [ ] Run `python scripts/run_demo.py` and verify V0 behavior remains functional.
- [ ] Run `python scripts/run_training_demo.py` and verify a complete finite OOS leaderboard.
- [ ] Scan source for API keys/secrets, network-only dependencies, live broker execution paths and random train/test split calls.
- [ ] Compare `feat/stockbot-v1-data-training` to `main` and summarize files/commits before PR creation.
