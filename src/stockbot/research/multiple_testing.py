from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DiscoveryResult:
    p_value: float
    q_value: float
    confidence: float
    observations: int
    significant: bool


def _newey_west_mean_standard_error(values: np.ndarray, max_lag: int | None = None) -> float:
    """Estimate the standard error of the sample mean with Bartlett HAC weights."""

    n = int(len(values))
    if n < 2:
        return float("inf")
    centered = values - float(np.mean(values))
    lag_cap = min(5, max(0, n // 4)) if max_lag is None else min(max_lag, n - 1)
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_variance = gamma0
    for lag in range(1, lag_cap + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (lag_cap + 1.0)
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(0.0, long_run_variance)
    return math.sqrt(long_run_variance / n)


def one_sided_mean_p_value(returns: pd.Series) -> tuple[float, int]:
    """Approximate one-sided positive-mean p-value using a HAC-adjusted normal tail.

    Financial return streams can be serially correlated. A naive iid standard error
    can therefore overstate significance. The bounded Newey-West/Bartlett adjustment
    makes the discovery gate more conservative when nearby OOS returns co-move.
    """

    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(r))
    if n < 3:
        return 1.0, n
    values = r.to_numpy(dtype=float)
    mean = float(np.mean(values))
    standard_error = _newey_west_mean_standard_error(values)
    if not math.isfinite(standard_error) or standard_error <= 0.0:
        return (0.0 if mean > 0 else 1.0), n
    statistic = mean / standard_error
    p = 0.5 * math.erfc(statistic / math.sqrt(2.0))
    return float(np.clip(p, 0.0, 1.0)), n


def benjamini_hochberg(p_values: list[float] | tuple[float, ...]) -> list[float]:
    """Return monotone Benjamini-Hochberg FDR-adjusted q-values."""

    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite and in [0,1]")

    order = np.argsort(values)
    ranked = values[order]
    m = len(values)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    q_values = np.empty_like(adjusted)
    q_values[order] = adjusted
    return [float(value) for value in q_values]


def evaluate_discoveries(
    returns_by_id: dict[str, pd.Series],
    *,
    max_q_value: float = 0.20,
) -> dict[str, DiscoveryResult]:
    """Apply HAC-aware significance tests and FDR control to a candidate population."""

    if not 0.0 < max_q_value < 1.0:
        raise ValueError("max_q_value must be in (0,1)")
    ids = list(returns_by_id)
    raw: list[tuple[float, int]] = [one_sided_mean_p_value(returns_by_id[item]) for item in ids]
    q_values = benjamini_hochberg([p for p, _ in raw])

    result: dict[str, DiscoveryResult] = {}
    for item, (p_value, observations), q_value in zip(ids, raw, q_values):
        confidence = float(np.clip(1.0 - q_value, 0.0, 1.0))
        result[item] = DiscoveryResult(
            p_value=float(p_value),
            q_value=float(q_value),
            confidence=confidence,
            observations=int(observations),
            significant=bool(q_value <= max_q_value),
        )
    return result
