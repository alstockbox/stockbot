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
