from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockbot.evaluation.metrics import performance_metrics


@dataclass(frozen=True)
class BootstrapInterval:
    lower: float
    median: float
    upper: float


@dataclass(frozen=True)
class BootstrapUncertaintyReport:
    observations: int
    block_size: int
    samples: int
    confidence_level: float
    cagr: BootstrapInterval
    sharpe: BootstrapInterval
    max_drawdown: BootstrapInterval
    probability_positive_cagr: float
    probability_positive_sharpe: float
    confidence_score: float


def _moving_block_sample(values: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=blocks)
    sampled: list[np.ndarray] = []
    for start in starts:
        indices = (np.arange(start, start + block_size) % n).astype(int)
        sampled.append(values[indices])
    return np.concatenate(sampled)[:n]


def _interval(values: np.ndarray, confidence_level: float) -> BootstrapInterval:
    alpha = (1.0 - confidence_level) / 2.0
    lower, median, upper = np.quantile(values, [alpha, 0.5, 1.0 - alpha])
    return BootstrapInterval(float(lower), float(median), float(upper))


def evaluate_block_bootstrap_uncertainty(
    returns: pd.Series,
    *,
    block_size: int = 21,
    samples: int = 500,
    confidence_level: float = 0.90,
    seed: int = 20260908,
) -> BootstrapUncertaintyReport:
    """Estimate uncertainty with a deterministic moving-block bootstrap.

    Contiguous return blocks are resampled rather than individual days so short-term
    serial dependence and volatility clustering are partially preserved. The result is
    a robustness diagnostic, not a guarantee or a classical iid confidence interval.
    """

    clean = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(clean)
    if n < 20:
        raise ValueError("at least 20 return observations are required")
    if block_size <= 0 or block_size > n:
        raise ValueError("block_size must be positive and no larger than observations")
    if samples < 50:
        raise ValueError("samples must be at least 50")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0.5,1)")

    values = clean.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    cagr_values = np.empty(samples, dtype=float)
    sharpe_values = np.empty(samples, dtype=float)
    drawdown_values = np.empty(samples, dtype=float)

    for i in range(samples):
        sample = _moving_block_sample(values, block_size, rng)
        sampled_series = pd.Series(sample, dtype=float)
        metrics = performance_metrics(sampled_series)
        cagr_values[i] = float(metrics.get("cagr", 0.0))
        sharpe_values[i] = float(metrics.get("sharpe", 0.0))
        drawdown_values[i] = float(metrics.get("max_drawdown", 0.0))

    probability_positive_cagr = float(np.mean(cagr_values > 0.0))
    probability_positive_sharpe = float(np.mean(sharpe_values > 0.0))
    cagr_interval = _interval(cagr_values, confidence_level)
    sharpe_interval = _interval(sharpe_values, confidence_level)
    drawdown_interval = _interval(drawdown_values, confidence_level)

    lower_tail_strength = float(
        np.clip(
            0.5 * (1.0 if cagr_interval.lower > 0.0 else 0.0)
            + 0.5 * (1.0 if sharpe_interval.lower > 0.0 else 0.0),
            0.0,
            1.0,
        )
    )
    confidence_score = float(
        np.clip(
            0.40 * probability_positive_cagr
            + 0.35 * probability_positive_sharpe
            + 0.25 * lower_tail_strength,
            0.0,
            1.0,
        )
    )

    return BootstrapUncertaintyReport(
        observations=n,
        block_size=int(block_size),
        samples=int(samples),
        confidence_level=float(confidence_level),
        cagr=cagr_interval,
        sharpe=sharpe_interval,
        max_drawdown=drawdown_interval,
        probability_positive_cagr=probability_positive_cagr,
        probability_positive_sharpe=probability_positive_sharpe,
        confidence_score=confidence_score,
    )
