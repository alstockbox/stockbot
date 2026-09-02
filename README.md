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
