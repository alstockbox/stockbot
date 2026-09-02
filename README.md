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
