# StockBot V0 Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable, testable StockBot V0 research/paper-trading core that can ingest market data, generate features and strategy signals, detect regimes, combine strategies, apply portfolio/risk rules, simulate costs/execution, evaluate performance, train baseline ML challengers, and generate structured AI/LLM research hypotheses.

**Architecture:** A provider-agnostic Python package with immutable domain objects and small modules. Strategies emit signals, portfolio construction emits target weights, risk gates those targets, and execution/backtesting consumes approved targets. ML and LLM components are challengers/research layers rather than unrestricted order generators.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, pydantic, scipy, pytest, pytest-cov, FastAPI (API milestone), optional yfinance only as a demo provider.

**Spec:** `docs/superpowers/specs/2026-09-02-stockbot-core-design.md`

## Global Constraints

- V0 is research, backtest, shadow and paper only; no live-money execution path.
- Initial tradable universe is liquid US equities and ETFs, long-only, no leverage, no options.
- Risk engine is authoritative and cannot be bypassed by ML, LLM or strategies.
- All feature and model calculations must be causal; no future data may enter a decision.
- Backtests include explicit commission and slippage assumptions.
- ML validation is time ordered; no random train/test split for trading performance claims.
- LLM output is structured research context/hypotheses, never a direct unrestricted BUY/SELL order.
- Deterministic seeds are used wherever stochastic algorithms are introduced.

---

### Task 1: Package foundation and domain contracts

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/stockbot/__init__.py`
- Create: `src/stockbot/domain/models.py`
- Test: `tests/test_domain_models.py`

**Interfaces:**
- Produces: `Signal`, `TargetPosition`, `RiskDecision`, `BacktestResult`, `MarketRegime` dataclasses/enums.

- [ ] Write tests that validate signal bounds, target-weight bounds, and risk-decision reason codes.
- [ ] Run the tests and confirm they fail because `stockbot.domain.models` does not exist.
- [ ] Implement minimal immutable domain types with validation.
- [ ] Run the focused test file and then full test suite.
- [ ] Commit as `feat: add stockbot domain contracts`.

### Task 2: Causal feature engine

**Files:**
- Create: `src/stockbot/features/technical.py`
- Create: `src/stockbot/features/pipeline.py`
- Test: `tests/features/test_technical_features.py`

**Interfaces:**
- Consumes: OHLCV pandas DataFrame indexed by timestamp.
- Produces: `build_technical_features(frame: pd.DataFrame) -> pd.DataFrame`.

- [ ] Write tests proving returns, momentum, moving-average distance, realized volatility and volume z-score use only current/past rows.
- [ ] Verify RED.
- [ ] Implement causal rolling/shifted features with explicit minimum periods and no backward filling.
- [ ] Verify GREEN and run full suite.
- [ ] Commit as `feat: add causal technical feature pipeline`.

### Task 3: Baseline strategy arena and regime detector

**Files:**
- Create: `src/stockbot/strategies/base.py`
- Create: `src/stockbot/strategies/baselines.py`
- Create: `src/stockbot/regimes/detector.py`
- Test: `tests/strategies/test_baseline_strategies.py`
- Test: `tests/regimes/test_detector.py`

**Interfaces:**
- Produces: `Strategy.generate_signal(features) -> float` in `[0,1]` for long-only V0.
- Produces: `detect_regime(features) -> MarketRegime`.

- [ ] Test trend, momentum, mean-reversion and breakout baselines on deterministic synthetic data.
- [ ] Test regime classification for bullish trend, bearish/stress and neutral/chop examples.
- [ ] Verify RED.
- [ ] Implement transparent baseline strategies and deterministic regime rules.
- [ ] Verify GREEN and full suite.
- [ ] Commit as `feat: add strategy arena baselines and regimes`.

### Task 4: Ensemble, portfolio construction and hard risk engine

**Files:**
- Create: `src/stockbot/ensemble/weighted.py`
- Create: `src/stockbot/portfolio/constructor.py`
- Create: `src/stockbot/risk/engine.py`
- Test: `tests/ensemble/test_weighted_ensemble.py`
- Test: `tests/risk/test_risk_engine.py`

**Interfaces:**
- Produces: `combine_signals(signals, regime_weights) -> float`.
- Produces: `target_from_signal(signal, volatility, config) -> TargetPosition`.
- Produces: `RiskEngine.evaluate(target, state) -> RiskDecision`.

- [ ] Test regime-weighted signal aggregation and bounded outputs.
- [ ] Test volatility-scaled sizing, max-position cap, daily-loss circuit breaker, drawdown state and stale-data rejection.
- [ ] Verify RED.
- [ ] Implement minimal ensemble, sizing and non-bypassable risk gate with reason codes.
- [ ] Verify GREEN and full suite.
- [ ] Commit as `feat: add portfolio and hard risk controls`.

### Task 5: Cost-aware event backtester and metrics

**Files:**
- Create: `src/stockbot/costs/model.py`
- Create: `src/stockbot/backtest/engine.py`
- Create: `src/stockbot/evaluation/metrics.py`
- Test: `tests/backtest/test_engine.py`
- Test: `tests/evaluation/test_metrics.py`

**Interfaces:**
- Produces: `LinearCostModel.estimate(notional, turnover) -> float`.
- Produces: `run_backtest(frame, signal_series, config) -> BacktestResult`.
- Produces: `performance_metrics(returns, benchmark_returns=None) -> dict[str,float]`.

- [ ] Test one-bar signal lag prevents same-bar lookahead.
- [ ] Test commission/slippage reduce P&L and turnover is charged.
- [ ] Test CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar and benchmark excess return.
- [ ] Verify RED.
- [ ] Implement cost-aware vector/event hybrid backtester and metrics.
- [ ] Verify GREEN and full suite.
- [ ] Commit as `feat: add cost-aware backtesting and evaluation`.

### Task 6: Machine-learning challenger pipeline

**Files:**
- Create: `src/stockbot/ml/dataset.py`
- Create: `src/stockbot/ml/challenger.py`
- Create: `src/stockbot/ml/validation.py`
- Test: `tests/ml/test_dataset.py`
- Test: `tests/ml/test_challenger.py`

**Interfaces:**
- Produces: `make_forward_return_dataset(features, close, horizon) -> (X, y)`.
- Produces: `walk_forward_splits(n_samples, train_size, test_size, embargo) -> list[tuple[np.ndarray,np.ndarray]]`.
- Produces: `MLChallenger.fit/predict_score` using deterministic scikit-learn baselines.

- [ ] Test labels are future returns but the final horizon rows are dropped.
- [ ] Test walk-forward splits are ordered and embargoed with no overlap.
- [ ] Test ML challenger output is finite and repeatable for fixed seed.
- [ ] Verify RED.
- [ ] Implement baseline HistGradientBoosting/regularized model challenger behind a stable interface.
- [ ] Verify GREEN and full suite.
- [ ] Commit as `feat: add machine learning challenger pipeline`.

### Task 7: Promotion scoring and champion/challenger registry

**Files:**
- Create: `src/stockbot/arena/scoring.py`
- Create: `src/stockbot/arena/registry.py`
- Test: `tests/arena/test_scoring.py`
- Test: `tests/arena/test_registry.py`

**Interfaces:**
- Produces: `research_score(metrics, robustness) -> float` penalizing drawdown, turnover, instability and tail risk.
- Produces: `ChampionRegistry.nominate()` and `promote_if_qualified()`.

- [ ] Test higher raw return does not win when produced with materially worse drawdown/cost/instability.
- [ ] Test challenger cannot promote without minimum OOS sample/robustness thresholds.
- [ ] Verify RED.
- [ ] Implement scoring and promotion gate.
- [ ] Verify GREEN and full suite.
- [ ] Commit as `feat: add champion challenger promotion gate`.

### Task 8: Structured LLM/AI research loop

**Files:**
- Create: `src/stockbot/llm/schemas.py`
- Create: `src/stockbot/ai/researcher.py`
- Create: `src/stockbot/learning/daily_review.py`
- Test: `tests/ai/test_researcher.py`

**Interfaces:**
- Produces: `ResearchHypothesis` Pydantic schema with hypothesis, evidence, proposed change, validation plan and invalidation criteria.
- Produces: `ResearchScientist.review(context) -> list[ResearchHypothesis]` using a provider interface.
- Produces: deterministic fallback rule-based hypotheses when no LLM provider is configured.

- [ ] Test malformed/free-form provider output is rejected.
- [ ] Test no hypothesis can directly request live order placement.
- [ ] Test deterministic fallback identifies cost, drawdown and regime degradation issues.
- [ ] Verify RED.
- [ ] Implement provider-agnostic structured LLM interface and daily-review fallback.
- [ ] Verify GREEN and full suite.
- [ ] Commit as `feat: add structured ai research loop`.

### Task 9: End-to-end research pipeline and CLI

**Files:**
- Create: `src/stockbot/research/pipeline.py`
- Create: `scripts/run_demo.py`
- Create: `tests/integration/test_research_pipeline.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `run_research(frame, benchmark=None) -> ResearchRun` with features, regime, baseline leaderboard, ensemble backtest and hypotheses.

- [ ] Test deterministic synthetic OHLCV can run through the entire pipeline and return finite metrics, an ensemble result and research hypotheses.
- [ ] Verify RED.
- [ ] Implement orchestrator with no network dependency.
- [ ] Verify GREEN and full suite with coverage.
- [ ] Document installation, demo execution, architecture and paper-only safety boundary.
- [ ] Commit as `feat: deliver stockbot v0 research pipeline`.

### Task 10: Verification and handoff

**Files:**
- Modify only if verification identifies defects.

- [ ] Run `python -m pip install -e '.[dev]'` in a clean environment.
- [ ] Run `pytest -q` and require zero failures.
- [ ] Run `python scripts/run_demo.py` and require a complete finite report.
- [ ] Inspect repository for accidental secrets and any live broker execution code.
- [ ] Compare feature branch with `main` and summarize all files/commits.
