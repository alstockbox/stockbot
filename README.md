# StockBot

StockBot is a research-first quantitative AI trading engine.

The project remains deliberately **research / backtest / paper / shadow only**. It contains causal features, deterministic ML challengers, champion/challenger scoring, point-in-time research inputs, sealed evidence boundaries, realistic execution diagnostics and a structured self-improving research loop.

It does **not** contain live broker order execution.

## Quick start

```bash
python -m pip install -e '.[dev]'
pytest -q
python scripts/run_demo.py
```

## Core principle

StockBot optimizes for repeatable out-of-sample edge after realistic costs, liquidity, market impact and risk constraints, not impressive in-sample P&L. Machine learning, ensembles, LLMs and research-memory signals are challenger/research layers. The hard risk engine is authoritative and cannot be bypassed by them.

See `docs/superpowers/specs/2026-09-02-stockbot-core-design.md` for the original architecture.

## V1: multi-symbol ML training

V1 adds provider-neutral cross-sectional research across many symbols:

- canonical market-data validation;
- point-in-time filtering;
- multi-symbol panels;
- causal cross-sectional features;
- multi-horizon forward-return/adverse-excursion labels;
- purged walk-forward validation with embargo;
- deterministic Ridge, ElasticNet, ExtraTrees, RandomForest and HistGradientBoosting challengers;
- reproducible dataset fingerprints;
- OOS-only Alpha Arena evaluation.

Run the deterministic training demo:

```bash
python scripts/run_training_demo.py
```

The bundled demo dataset is explicitly non-research-grade. A model can rank first on demo data, but the promotion gate will not treat it as a valid research-grade champion candidate. Trading-performance evaluation is assembled from time-ordered out-of-sample predictions; random train/test splits are not used for performance claims.

## V1.2: immutable real-market snapshots

V1.2 connects the training engine to historical EOD data through provider adapters and immutable snapshots.

### Tiingo

```bash
export TIINGO_API_TOKEN="your-token"
python scripts/run_market_training.py \
  --provider tiingo \
  --symbols AAPL,MSFT,NVDA,SPY \
  --start 2018-01-01 \
  --end 2026-09-01 \
  --snapshot-root snapshots \
  --train
```

Tiingo snapshots preserve raw and adjusted OHLCV plus dividend and split fields. EOD-only downloads default to `BOOTSTRAP`; using a serious provider alone does not make a today's-universe backtest survivorship-bias-free or point-in-time research-grade.

### Zero-key bootstrap prices

```bash
python scripts/run_market_training.py \
  --provider yahoo-bootstrap \
  --symbols AAPL,MSFT,NVDA,SPY \
  --start 2020-01-01 \
  --end 2026-09-01 \
  --snapshot-root snapshots \
  --train
```

The Yahoo chart adapter is intentionally labeled `BOOTSTRAP`. Each download produces an immutable snapshot containing market bars plus a manifest with provider, data grade, universe, dates, row count and content fingerprint. Credentials are excluded from manifests and fingerprints.

## V2: autonomous Research Factory

V2 expands the Alpha Arena into a larger autonomous research system while preserving strict evidence and risk boundaries.

The factory includes:

- balanced deterministic challenger populations across five model families;
- default horizons of 1, 5 and 20 trading days;
- parallel evaluation while keeping each estimator deterministic;
- risk-adjusted objectives incorporating CAGR, Sharpe, Sortino, Calmar, excess return, drawdown, CVaR, volatility, turnover, instability and concentration;
- adversarial stress tests;
- strict research-grade/OOS/robustness/risk/stress gates;
- append-only experiment memory;
- hard-negative mining;
- a blind internal holdout reserved before model search;
- separate horizon champions and a global champion/challenger path;
- correlation-aware horizon ensembles.

Example:

```bash
python scripts/run_market_training.py \
  --provider tiingo \
  --symbols AAPL,MSFT,NVDA,AMZN,META,GOOGL,SPY,QQQ \
  --start 2014-01-01 \
  --end 2026-09-01 \
  --snapshot-root snapshots \
  --factory \
  --factory-horizons 1,5,20 \
  --factory-candidates 160 \
  --factory-workers 4 \
  --factory-memory research_memory/experiments.jsonl
```

A BOOTSTRAP snapshot can exercise and rank the research machinery but cannot become a valid research-grade champion source.

### What V2 optimizes

V2 is intentionally **not** trained against a fixed weekly-return target. The factory searches for robust risk-adjusted out-of-sample edge after costs and then subjects the strongest candidates to independent evidence gates. Strict gates are allowed to produce no champion.

## Evidence chain

The intended research funnel is:

`broad/evolutionary population -> purged OOS -> realistic costs -> adversarial stress -> regime evaluation -> FDR control -> drift checks -> internal blind holdout -> holdout-safe Deep Research -> sealed quarantine single audit -> forward paper arena -> manual live review`

The blind holdout and sealed quarantine are not optimization datasets.

### Statistical false-discovery control

When many candidates are tested, some will look good by chance. StockBot applies one-sided OOS return testing with Newey-West/HAC uncertainty and Benjamini-Hochberg false-discovery-rate control across the experiment population.

### Regime robustness and specialists

Generalists are evaluated across causal `bull_trend`, `bear_stress` and `neutral_chop` regimes. Optional specialists learn and predict only inside their intended regime, and a causal research-only router can combine their OOS return streams. Specialist/router output remains diagnostic and cannot bypass the global evidence chain.

### Evolutionary challenger generation

A bounded portion of future populations can be generated by mutating historically strong research-gate winners while the rest of the population continues broad exploration. This gives the factory an explore/exploit loop without turning past winners into automatic promotions.

## Scheduler-friendly research jobs

Factory runs can emit deterministic manifests and machine-readable artifacts:

```bash
python scripts/run_market_training.py \
  --provider tiingo \
  --symbols AAPL,MSFT,NVDA,AMZN,META,GOOGL,SPY,QQQ \
  --start 2014-01-01 \
  --end 2026-09-01 \
  --snapshot-root snapshots \
  --factory \
  --factory-memory research_memory/experiments.jsonl \
  --factory-run-dir research_runs \
  --factory-regime-specialists
```

A run directory can contain:

- `job.json`: scheduler/job identity including search/execution configuration and evidence fingerprints;
- `research_cycle.json`: data-evidence cycle identity plus operational Deep Findings settings;
- `data_quality.json`: fail-closed research-data evidence;
- `summary.json`: compact factory/specialist outcome;
- `deep_diagnostics.json`: compact expensive diagnostics when enabled;
- `quarantine.json`: sealed-quarantine metadata when configured.

`job.json` and `research_cycle.json` intentionally answer different questions. Job identity may vary when worker count or candidate budget changes. `research_cycle.json` keeps the same `research_cycle_id` when the underlying evidence is unchanged, preventing a rerun with different execution knobs from masquerading as a new data cycle.

## Deep diagnostics

Expensive diagnostics are restricted to the pre-holdout research partition and include:

- policy search;
- moving-block bootstrap uncertainty;
- factor exposure and residual-edge attribution;
- liquidity/market-impact simulation;
- capital-capacity curves;
- multi-window robustness;
- feature-group and auxiliary-feature ablation;
- OOS-only stacking/meta-model diagnostics;
- point-in-time sector/factor neutralization;
- execution-aware point-in-time sector-cap challengers.

Compact diagnostics retain useful metrics while omitting heavy OOS return streams and full portfolio-weight matrices.

## Point-in-time external signals

External fundamentals, macro, rates, FX, commodities, news/sentiment, short interest, insider and alternative-data signals use `PointInTimeFeatureStore`. Every observation distinguishes economic/event `observation_time` from earliest strategy-usable `available_time`. Missing timing is rejected rather than guessed. Research-input files are fingerprinted and verified on reload, so changed revisions change research identity.

### Riksbank historical macro vintages

StockBot includes a fail-closed adapter for Sveriges Riksbank Monetary Policy Data API. The initial feature set includes policy rate, CPIF, CPI, calendar-adjusted GDP growth, unemployment and KIX.

```bash
python scripts/build_riksbank_macro_store.py \
  --output research_inputs/riksbank_macro.json \
  --series policy_rate,cpif_yoy,gdp_yoy_ca,unemployment_rate,kix_index \
  --start-year 2020
```

Then pass the fingerprinted store to the factory with `--factory-auxiliary-input` and an optional feature subset.

## Fail-closed research-grade data

A declared `RESEARCH_GRADE` label is not trusted by itself. `ResearchDataQualityReport` verifies observable adjusted-market-data quality, retrieval-time causality, provider/corporate-action fields and point-in-time universe coverage. Explicit attestations are required for properties that cannot be inferred from columns alone.

If required evidence is absent or fails, a declared research-grade snapshot is downgraded to `BOOTSTRAP`. A passing quality report never upgrades an already lower declared grade.

`PointInTimeUniverse` stores historical membership intervals plus optional exchange, sector, industry and delisting metadata. Serious survivorship-control evidence requires point-in-time membership semantics and delisted-security coverage.

## Sealed quarantine and single audit

The final quarantine boundary is an explicit fixed date. It does not silently roll forward as new observations arrive. Future rows at or after that date remain quarantined until an explicit new research cycle is created.

Routine factory, specialist and deep-diagnostic jobs operate on development data only. The quarantine audit permits one frozen strategy specification per sealed cycle. Re-testing variants against the same quarantine is rejected so the quarantine cannot become another hyperparameter leaderboard.

## Forward paper arena and deployment evidence

The paper ledger is append-only. The paper arena evaluates forward sessions, calendar span, excess performance, turnover, cost rate, fill quality, stress, drift and moving-block-bootstrap confidence.

A deployment-evidence gate can require all of:

1. research readiness;
2. verified `RESEARCH_GRADE` evidence;
3. a passed sealed-quarantine audit;
4. sufficient forward paper evidence.

Even if all conditions pass, the result is only **eligible for manual live review**. `hard_risk_engine_required=True` and `broker_execution_available=False`.

## Deep Findings: future-cycle learning without same-cycle leakage

Deep Research compacts development-only diagnostics into append-only `DeepResearchFinding` records. A bounded `adaptive_priority` combines normalized bootstrap confidence, idiosyncratic factor score, capacity score and multi-window robustness. Descriptive score deltas are retained for analysis but cannot dominate future search allocation.

A deterministic `research_cycle_id` is derived from evidence identity:

- market dataset fingerprint;
- sealed-quarantine start;
- point-in-time universe fingerprint;
- auxiliary feature-store fingerprint;
- research data-quality fingerprint.

Worker count, candidate budget and other execution knobs are deliberately excluded. Re-running the same underlying evidence is therefore the **same cycle**.

`ResearchCycleManifest` / `research_cycle.json` makes this boundary auditable. It records the five evidence-identity fields together with `deep_feedback_path` and bounded `deep_feedback_weight`. The path and weight are operational metadata and do not alter `research_cycle_id`.

Deep Findings from the current cycle are ignored when ranking mutation parents. Only findings from earlier data cycles may provide a bounded bonus to future exploration allocation. They never modify research gates, factory score, blind-holdout score, promotion score or champion eligibility.

The CLI stores Deep Findings beside experiment memory as `deep-findings.jsonl` and writes the cycle manifest atomically before factory execution when a run directory is configured.

Example:

```bash
python scripts/run_market_training.py \
  --provider tiingo \
  --symbols AAPL,MSFT,NVDA,AMZN,META,GOOGL,SPY,QQQ \
  --start 2014-01-01 \
  --end 2026-09-01 \
  --snapshot-root snapshots \
  --factory \
  --factory-memory research_memory/experiments.jsonl \
  --factory-run-dir research_runs \
  --factory-universe-input research_inputs/universe.json \
  --factory-attest-adjusted-prices-verified \
  --factory-attest-corporate-actions-complete \
  --factory-attest-corporate-actions-point-in-time \
  --factory-quarantine-start 2026-01-02 \
  --factory-deep-diagnostics \
  --factory-deep-feedback-weight 0.25
```

Attestations are evidence declarations, not a way to manufacture research-grade status: observable data/universe checks still fail closed and the declared snapshot grade is never automatically upgraded.

## Safety boundary

No backtest, model score, statistical test, champion status, quarantine result, paper result or deployment-evidence result guarantees future returns.

StockBot currently provides research, backtesting, diagnostics, sealed evaluation and paper/shadow evidence only. There is **no live broker execution route** in this repository.
