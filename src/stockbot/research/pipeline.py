from __future__ import annotations

import math

import numpy as np
import pandas as pd

from stockbot.ai.researcher import ResearchScientist
from stockbot.arena.registry import ChampionRegistry, PromotionCriteria
from stockbot.arena.scoring import research_score
from stockbot.backtest.engine import BacktestConfig, run_backtest
from stockbot.domain.models import MarketRegime, ResearchRun
from stockbot.ensemble.weighted import combine_signals
from stockbot.features.pipeline import build_technical_features
from stockbot.ml.dataset import make_forward_return_dataset
from stockbot.ml.walkforward import walk_forward_predictions
from stockbot.portfolio.constructor import PortfolioConfig, target_from_signal
from stockbot.regimes.detector import detect_regime
from stockbot.risk.engine import RiskConfig, RiskEngine
from stockbot.strategies.baselines import BASELINE_STRATEGIES


REGIME_WEIGHTS: dict[MarketRegime, dict[str, float]] = {
    MarketRegime.BULL_TREND: {
        "trend": 0.30,
        "momentum": 0.30,
        "mean_reversion": 0.08,
        "breakout": 0.17,
        "ml_ridge": 0.15,
    },
    MarketRegime.NEUTRAL_CHOP: {
        "trend": 0.12,
        "momentum": 0.12,
        "mean_reversion": 0.40,
        "breakout": 0.12,
        "ml_ridge": 0.14,
    },
    MarketRegime.BEAR_STRESS: {
        "trend": 0.04,
        "momentum": 0.04,
        "mean_reversion": 0.12,
        "breakout": 0.04,
        "ml_ridge": 0.06,
    },
}


def _signal_series(features: pd.DataFrame, strategy) -> pd.Series:
    values = [strategy.generate_signal(row) for _, row in features.iterrows()]
    return pd.Series(values, index=features.index, dtype=float)


def _raw_prediction_to_long_score(prediction: pd.Series, scale: float = 0.015) -> pd.Series:
    z = (prediction / scale).clip(-20.0, 20.0)
    score = 1.0 / (1.0 + np.exp(-z))
    return score.where(prediction.notna(), 0.0)


def _build_ml_oos_signal(features: pd.DataFrame, close: pd.Series) -> tuple[pd.Series, int]:
    signal = pd.Series(0.0, index=features.index, dtype=float)
    X, y = make_forward_return_dataset(features, close, horizon=5)
    if len(X) < 170:
        return signal, 0

    train_size = min(126, max(80, len(X) // 2))
    test_size = 21
    embargo = 5
    predictions = walk_forward_predictions(
        X,
        y,
        train_size=train_size,
        test_size=test_size,
        embargo=embargo,
        model_type="ridge",
        seed=7,
    )
    scored = _raw_prediction_to_long_score(predictions)
    signal.loc[scored.index] = scored
    return signal, int(predictions.notna().sum())


def run_research(
    frame: pd.DataFrame,
    benchmark: pd.Series | None = None,
    symbol: str = "ASSET",
) -> ResearchRun:
    features = build_technical_features(frame)
    benchmark_returns = benchmark.pct_change().fillna(0.0) if benchmark is not None else None
    backtest_cfg = BacktestConfig()
    risk_engine = RiskEngine(
        RiskConfig(
            max_position_weight=1.0,
            max_drawdown=0.10,
            max_daily_loss=0.04,
            max_data_age_seconds=300.0,
            abnormal_volatility=0.80,
        )
    )

    strategy_signals: dict[str, pd.Series] = {}
    leaderboard: list[dict[str, float | str]] = []
    robustness_by_name: dict[str, float] = {}
    samples_by_name: dict[str, int] = {}

    for strategy in BASELINE_STRATEGIES:
        signals = _signal_series(features, strategy).fillna(0.0)
        strategy_signals[strategy.name] = signals
        bt = run_backtest(
            frame,
            signals,
            backtest_cfg,
            benchmark_returns,
            risk_engine=risk_engine,
        )
        robustness = max(0.0, min(1.0, 1.0 - bt.metrics["max_drawdown"]))
        score = research_score(bt.metrics, robustness)
        leaderboard.append({"name": strategy.name, "score": score, **bt.metrics})
        robustness_by_name[strategy.name] = robustness
        samples_by_name[strategy.name] = max(0, len(frame) - 21)

    ml_signal, ml_oos_samples = _build_ml_oos_signal(features, frame["close"])
    if ml_oos_samples > 0:
        strategy_signals["ml_ridge"] = ml_signal
        ml_bt = run_backtest(
            frame,
            ml_signal,
            backtest_cfg,
            benchmark_returns,
            risk_engine=risk_engine,
        )
        coverage = min(1.0, ml_oos_samples / max(1, len(frame) // 2))
        ml_robustness = max(0.0, min(1.0, (1.0 - ml_bt.metrics["max_drawdown"]) * coverage))
        ml_score = research_score(ml_bt.metrics, ml_robustness)
        leaderboard.append({"name": "ml_ridge", "score": ml_score, **ml_bt.metrics})
        robustness_by_name["ml_ridge"] = ml_robustness
        samples_by_name["ml_ridge"] = ml_oos_samples
    else:
        strategy_signals["ml_ridge"] = ml_signal

    leaderboard.sort(key=lambda row: float(row["score"]), reverse=True)

    registry = ChampionRegistry(
        PromotionCriteria(min_oos_samples=50, min_robustness=0.40, score_margin=0.02, max_drawdown=0.50)
    )
    for row in leaderboard:
        name = str(row["name"])
        registry.nominate(
            name,
            score=float(row["score"]),
            metrics={k: float(v) for k, v in row.items() if k not in {"name"}},
            robustness=robustness_by_name[name],
            oos_samples=samples_by_name[name],
        )
        registry.promote_if_qualified(name)
    champion_name = registry.champion.name if registry.champion else None

    regime_series = features.apply(detect_regime, axis=1)
    raw_ensemble: list[float] = []
    portfolio_cfg = PortfolioConfig(target_volatility=0.18, max_position_weight=1.0)

    for idx, row in features.iterrows():
        signals = {name: float(series.loc[idx]) for name, series in strategy_signals.items()}
        regime = regime_series.loc[idx]
        combined = combine_signals(signals, REGIME_WEIGHTS[regime])
        vol = row.get("realized_vol_20", np.nan)
        try:
            vol_float = float(vol)
        except (TypeError, ValueError):
            vol_float = math.nan
        target = target_from_signal(symbol, combined, vol_float, portfolio_cfg)
        raw_ensemble.append(target.weight)

    ensemble_signal = pd.Series(raw_ensemble, index=frame.index, dtype=float).fillna(0.0)
    ensemble_bt = run_backtest(
        frame,
        ensemble_signal,
        backtest_cfg,
        benchmark_returns,
        risk_engine=risk_engine,
    )
    latest_regime = regime_series.iloc[-1]
    latest_signal = float(ensemble_signal.iloc[-1])

    context = {
        "metrics": ensemble_bt.metrics,
        "regime_degradation": latest_regime is MarketRegime.BEAR_STRESS and ensemble_bt.metrics["cagr"] < 0,
        "leaderboard": leaderboard,
        "champion_name": champion_name,
        "ml_oos_samples": ml_oos_samples,
    }
    hypotheses = ResearchScientist().review(context)

    return ResearchRun(
        leaderboard=leaderboard,
        ensemble=ensemble_bt,
        latest_regime=latest_regime,
        latest_signal=latest_signal,
        hypotheses=hypotheses,
        champion_name=champion_name,
        ml_oos_samples=ml_oos_samples,
    )
