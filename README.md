# StockBot

StockBot is a research-first quantitative AI trading engine.

V0 is deliberately **paper/research only**. It contains causal features, transparent baseline strategies,
market-regime detection, portfolio sizing, hard risk gates, cost-aware backtesting, deterministic ML
challengers, champion/challenger scoring, and a structured AI/LLM research loop.

It does **not** contain live broker order execution.

## Quick start

```bash
python -m pip install -e '.[dev]'
pytest -q
python scripts/run_demo.py
```

## Core principle

The system optimizes for repeatable out-of-sample edge after costs, not impressive in-sample P&L.
Machine learning and LLMs are research/challenger layers. The risk engine is authoritative and cannot
be bypassed.

See `docs/superpowers/specs/2026-09-02-stockbot-core-design.md` for the architecture.

## V1: multi-symbol ML training

V1 adds a provider-neutral training layer for cross-sectional research across many symbols. It includes canonical market-data validation, point-in-time `available_time` filtering, multi-symbol panel construction, causal cross-sectional features, multi-horizon forward-return/adverse-excursion labels, purged walk-forward validation with embargo, a deterministic five-model ML zoo, reproducible dataset fingerprints, and an OOS-only Alpha Arena.

Run the deterministic training demo:

```bash
python scripts/run_training_demo.py
```

The bundled demo dataset is explicitly `DEMO / NON-RESEARCH-GRADE`. A model can rank first on demo data, but the promotion gate will not treat it as a valid champion candidate. Research-grade promotion requires a dataset explicitly marked `DataGrade.RESEARCH_GRADE` and still must pass OOS coverage, robustness, drawdown, turnover and score gates.

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

V2.1 adds another evidence layer between raw model search and extended paper trading. The default selection funnel is now:

`broad/evolutionary population -> purged OOS -> cost-aware metrics -> adversarial stress -> regime evaluation -> FDR control -> drift check -> blind holdout -> paper readiness -> horizon champions -> meta-ensemble`

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

Research memory is now used as an input to future snapshot runs. A bounded portion of the next population is created by mutating parameters around historically strong research-gate winners, while the remaining population continues broad model-family exploration. This gives the search loop an explore/exploit mechanism instead of restarting from exactly the same static grid every run.

### Drift and paper-readiness

Recent OOS behavior is compared with each model's earlier OOS history using mean-return shift, volatility expansion, Sharpe degradation and drawdown degradation. A final paper-readiness report combines OOS coverage, robustness, stress survival, regime robustness, FDR confidence, drift stability and blind-holdout evidence. By default, a model is not eligible for champion promotion unless the evidence chain is complete.

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

Each run can write `job.json` before research begins and `summary.json` after it completes. The summary includes experiment counts, promotion status, active champion, horizon-ensemble score and optional regime-specialist/router scores. This is intended as the persistence contract for a future scheduler/queue rather than as a live execution service.

No backtest, model score, statistical test, paper-readiness status or champion status guarantees future returns.
