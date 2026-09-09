# StockBot

StockBot is a research-first quantitative AI trading engine.

The project remains deliberately **paper/research only**. It contains causal features, transparent baseline strategies,
market-regime detection, portfolio sizing, hard risk gates, cost-aware backtesting, deterministic ML
challengers, champion/challenger scoring, point-in-time research inputs, sealed evidence boundaries and a structured AI/ML research loop.

It does **not** contain live broker order execution.

## Quick start

```bash
python -m pip install -e '.[dev]'
pytest -q
python scripts/run_demo.py
```

## Core principle

The system optimizes for repeatable out-of-sample edge after realistic costs, liquidity, market impact and risk constraints, not impressive in-sample P&L.
Machine learning and LLMs are research/challenger layers. The hard risk engine is authoritative and cannot
be bypassed by a model, an ensemble, an LLM or a research-memory signal.

See `docs/superpowers/specs/2026-09-02-stockbot-core-design.md` for the original architecture.

## V1: multi-symbol ML training

V1 adds a provider-neutral training layer for cross-sectional research across many symbols. It includes canonical market-data validation, point-in-time `available_time` filtering, multi-symbol panel construction, causal cross-sectional features, multi-horizon forward-return/adverse-excursion labels, purged walk-forward validation with embargo, a deterministic five-model ML zoo, reproducible dataset fingerprints, and an OOS-only Alpha Arena.

Run the deterministic training demo:

```bash
python scripts/run_training_demo.py
```

The bundled demo dataset is explicitly `DEMO / NON-RESEARCH-GRADE`. A model can rank first on demo data, but the promotion gate will not treat it as a valid champion candidate. Research-grade promotion requires verified research-grade evidence and still must pass OOS coverage, robustness, drawdown, turnover and score gates.

### V1 model families

- Ridge
- ElasticNet
- ExtraTrees
- RandomForest
- HistGradientBoosting

All trading-performance evaluation is assembled from time-ordered out-of-sample predictions. Random train/test splits are not used for performance claims.

## V1.2: real market data bootstrap

V1.2 connects the V1 training engine to real historical EOD data through provider adapters and immutable snapshots.

### Tiingo (preferred serious EOD provider)

Set your token locally; never commit it:

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

Tiingo snapshots preserve both raw and adjusted OHLCV plus dividends and split factors. EOD-only downloads default to `BOOTSTRAP` grade; merely using Tiingo does **not** automatically make a today's-universe backtest survivorship-bias-free or point-in-time research-grade.

### Zero-key bootstrap prices

For immediate experiments without an API key:

```bash
python scripts/run_market_training.py \
  --provider yahoo-bootstrap \
  --symbols AAPL,MSFT,NVDA,SPY \
  --start 2020-01-01 \
  --end 2026-09-01 \
  --snapshot-root snapshots \
  --train
```

The Yahoo chart adapter is intentionally labeled `BOOTSTRAP`: it is an unofficial data path and has no historical-universe / point-in-time guarantee. Alpha Arena can rank models on these snapshots, but the promotion gate can never treat them as a valid champion source.

Each download produces an immutable directory containing `bars.csv` and `manifest.json` with the exact universe, date range, provider, data grade, row count and content fingerprint. API credentials are excluded from manifests and fingerprints.

## V2: autonomous Research Factory

V2 turns the existing Alpha Arena into a much larger autonomous research engine. It does **not** remove the hard risk architecture or enable live broker execution. Its purpose is to search much more aggressively for repeatable edge while making false discoveries harder to promote.

The V2 factory adds:

- a balanced population of up to 160 deterministic ML challengers **per horizon** across Ridge, ElasticNet, ExtraTrees, RandomForest and HistGradientBoosting;
- parallel challenger evaluation while keeping each individual estimator deterministic;
- default multi-horizon competition at 1, 5 and 20 trading days;
- a stronger risk-adjusted objective that rewards CAGR, Sharpe, Sortino, Calmar, excess return, OOS coverage and robustness while penalizing drawdown, CVaR, volatility, turnover, instability and concentration;
- candidate-specific adversarial stress tests against amplified losses, higher trading friction, periodic gap shocks and combined adverse conditions;
- strict research-grade, OOS, robustness, drawdown, CVaR, turnover, instability, concentration and stress promotion gates;
- an append-only cached research memory so failed and successful experiments remain auditable without repeatedly rereading the entire experiment history;
- a hard-negative miner for high-confidence directional mistakes;
- a true blind final holdout reserved **before** hyperparameter search. Only the strongest gated candidates are evaluated on it, and holdout passage is required for champion promotion;
- separate horizon champions plus a global champion/challenger promotion path.

Run the factory on an immutable market snapshot:

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

A BOOTSTRAP snapshot can be used to exercise and rank the research machinery, but it still cannot become a valid research-grade champion source. For serious promotion claims, the input universe and market data must be point-in-time and survivorship-bias controlled.

### What V2 optimizes

V2 is intentionally **not** trained against a fixed weekly-return target. Searching hundreds of candidates against an arbitrary return target is a direct route to curve fitting and hidden leverage. The factory instead maximizes robust out-of-sample edge after costs, then subjects the strongest candidates to stress and blind-holdout tests before promotion.

## V2.1: self-improving research farm

V2.1 adds another evidence layer between raw model search and extended paper trading. The research funnel now combines:

`broad/evolutionary population -> purged OOS -> realistic costs -> adversarial stress -> regime evaluation -> FDR control -> drift checks -> internal blind holdout -> holdout-safe Deep Research -> sealed quarantine single audit -> forward paper arena -> manual live review`

Deep Research is invoked after the factory report exists, but its expensive diagnostics replay only the pre-holdout research partition. The internal blind holdout and the later sealed quarantine are not optimization datasets.

### Statistical false-discovery control

When hundreds of candidates are tested, some will look good by luck. V2.1 computes a one-sided OOS mean-return test for each candidate and applies Benjamini-Hochberg false-discovery-rate control across the whole experiment population. A candidate can therefore have an attractive backtest score and still be rejected before it consumes blind-holdout budget.

### Regime robustness and specialists

Every generalist is evaluated separately across causal `bull_trend`, `bear_stress` and `neutral_chop` market regimes. V2.1 can also train dedicated purged-walk-forward specialists that only learn and predict in their intended regime, then evaluate a causal research-only regime router across their OOS return streams.

Run the optional specialist research after the main factory:

```bash
python scripts/run_market_training.py \
  --provider tiingo \
  --symbols AAPL,MSFT,NVDA,AMZN,META,GOOGL,SPY,QQQ \
  --start 2014-01-01 \
  --end 2026-09-01 \
  --snapshot-root snapshots \
  --factory \
  --factory-regime-specialists \
  --factory-specialists-top-k 2
```

Regime specialists and the router remain diagnostics. They do not bypass the global blind holdout or create a broker execution path.

### Evolutionary challenger generation

Research memory is used as an input to future snapshot runs. A bounded portion of the next population is created by mutating parameters around historically strong research-gate winners, while the remaining population continues broad model-family exploration. This gives the search loop an explore/exploit mechanism instead of restarting from exactly the same static grid every run.

### Drift and paper-readiness

Recent OOS behavior is compared with each model's earlier OOS history using mean-return shift, volatility expansion, Sharpe degradation and drawdown degradation. A paper-readiness report combines OOS coverage, robustness, stress survival, regime robustness, FDR confidence, drift stability and blind-holdout evidence. Paper-readiness is evidence, not permission to place orders.

### Scheduler-friendly 24/7 research jobs

Factory runs can emit deterministic job manifests and machine-readable summaries:

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

Each run can write `job.json` before research begins and `summary.json` after it completes. Job identity binds the market snapshot, sealed-quarantine boundary, point-in-time auxiliary-data fingerprint, selected auxiliary features, historical-universe fingerprint and data-quality fingerprint. The summary includes experiment counts, promotion status, active champion, horizon-ensemble score and optional regime-specialist/router scores. This is the persistence contract for a scheduler/queue, not a live execution service.

## V2.2: deep diagnostics and point-in-time external signals

The research stack retains and replays the exact feature contract used by each candidate. Expensive diagnostics are restricted to the pre-holdout research partition and include policy search, moving-block bootstrap uncertainty, factor attribution, liquidity/market-impact simulation, capital-capacity curves, multi-window robustness, feature-group ablation, OOS stacking, point-in-time sector/factor neutralization and a point-in-time sector-cap challenger.

External fundamentals, macro, news, sentiment, short-interest, insider and alternative-data signals enter through `PointInTimeFeatureStore`. Every observation requires both an economic/event `observation_time` and the earliest strategy-usable `available_time`. Missing publication timing is rejected rather than guessed. Research-input files are fingerprinted and verified on reload, so changed revisions create a different research identity.

### Riksbank historical macro vintages

StockBot includes a fail-closed adapter for Sveriges Riksbank Monetary Policy Data API. The initial stable feature set includes policy rate, CPIF, CPI, calendar-adjusted GDP growth, unemployment and the KIX exchange-rate index. The historical-vintage collector replays policy rounds from 2020 onward. V1 of this adapter materializes realised observations only; future forecast targets are intentionally kept out until they have an explicit horizon-aware feature schema.

Build a fingerprinted Riksbank vintage store:

```bash
python scripts/build_riksbank_macro_store.py \
  --output research_inputs/riksbank_macro.json \
  --series policy_rate,cpif_yoy,gdp_yoy_ca,unemployment_rate,kix_index \
  --start-year 2020
```

Then pass it into the factory:

```bash
python scripts/run_market_training.py \
  --provider tiingo \
  --symbols AAPL,MSFT,NVDA,AMZN,META,GOOGL,SPY,QQQ \
  --start 2014-01-01 \
  --end 2026-09-01 \
  --snapshot-root snapshots \
  --factory \
  --factory-auxiliary-input research_inputs/riksbank_macro.json \
  --factory-auxiliary-features policy_rate,cpif_yoy,gdp_yoy_ca,unemployment_rate,kix_index \
  --factory-deep-diagnostics
```

When a candidate actually uses `aux__*` features, Deep Research additionally runs auxiliary ablation: full external-signal stack versus no external data and per-feature leave-one-out tests. These diagnostics report aggregate external-data contribution, useful features and potentially harmful features. They remain diagnostics and do not replace blind holdout, sealed quarantine or forward paper evidence.

## V2.3: sealed evidence chain and future-cycle Deep Findings

### Fail-closed research-grade data

A declared `RESEARCH_GRADE` label is not trusted by itself. `ResearchDataQualityReport` verifies observable adjusted-market-data quality, retrieval-time causality, provider/corporate-action fields and point-in-time universe coverage. Explicit attestations are required for properties that cannot be inferred from columns alone, including adjusted-price verification and corporate-action completeness/point-in-time correctness.

If the evidence is absent or fails, the effective grade is downgraded to `BOOTSTRAP`; a passing report never upgrades an already lower declared grade.

`PointInTimeUniverse` stores historical membership intervals plus optional exchange, sector, industry and delisting metadata. Research-grade universe evidence requires point-in-time membership semantics, survivorship-bias control and delisted-security coverage.

### Sector exposure controls

Deep Research can measure point-in-time sector exposure, evaluate factor/sector neutralization and test an execution-aware sector-cap challenger. Sector constraints are diagnostics/challengers: they do not retroactively change the blind holdout and cannot override the hard risk engine.

Compact deep-diagnostic JSON retains score/Sharpe/CAGR/stress and sector concentration before/after, while omitting heavy OOS return streams and full weight matrices.

### Sealed quarantine and single audit

The final quarantine boundary is an explicit fixed date. It does not silently roll forward with the newest data. Future observations at or after that date remain quarantined until a new research cycle is explicitly created.

Routine factory, specialist and deep-diagnostic jobs operate on development data only. The quarantine audit permits one frozen strategy specification per sealed research cycle and records the result in an append-only ledger. Re-testing variants against the same quarantine is rejected so the quarantine cannot turn into another hyperparameter leaderboard.

### Forward paper arena and deployment evidence

The paper ledger is append-only and the paper arena evaluates forward sessions, calendar span, excess performance, turnover, cost rate, fill quality, stress, drift and moving-block-bootstrap confidence.

A deployment-evidence gate can require all of the following:

1. research readiness;
2. verified `RESEARCH_GRADE` evidence;
3. a passed sealed-quarantine audit;
4. sufficient forward paper evidence.

Even if all conditions pass, the result is only **eligible for manual live review**. `hard_risk_engine_required=True` and `broker_execution_available=False`; StockBot still cannot submit a live order.

### Deep Findings: learning without same-cycle leakage

Deep Research compacts development-only diagnostics into append-only `DeepResearchFinding` records. Findings capture bounded signals such as bootstrap confidence, idiosyncratic factor score, capacity score and multi-window robustness, plus descriptive policy/ablation/sector findings.

A deterministic `research_cycle_id` is derived from data identity:

- market dataset fingerprint;
- sealed-quarantine start;
- point-in-time universe fingerprint;
- auxiliary feature-store fingerprint;
- research data-quality fingerprint.

Execution knobs such as worker count or candidate budget are deliberately excluded. Re-running the same underlying evidence is therefore the **same cycle**.

Deep Findings from the current cycle are ignored when ranking mutation parents. Only findings from earlier data cycles may provide a bounded bonus to future exploration allocation. They never modify research gates, factory score, blind-holdout score, promotion score or champion eligibility.

The CLI stores Deep Findings beside experiment memory as `deep-findings.jsonl` and prints both `research_cycle_id` and `deep_feedback_path`. The feedback strength is bounded by `--factory-deep-feedback-weight` in `[0,1]`.

Example research cycle:

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

The attestations above are evidence declarations, not a way to manufacture research-grade status: observable data/universe checks still fail closed and the declared snapshot grade is never automatically upgraded.

## Safety boundary

No backtest, model score, statistical test, paper-readiness status, champion status, auxiliary-data result, quarantine result or paper-arena result guarantees future returns.

StockBot currently provides research, backtesting, diagnostics, sealed evaluation and paper/shadow evidence only. There is **no live broker execution route** in this repository.
