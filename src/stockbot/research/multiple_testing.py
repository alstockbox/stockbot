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


def one_sided_mean_p_value(returns: pd.Series) -> tuple[float, int]:
    """Approximate one-sided p-value for positive mean return using a normal tail."""

    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(r))
    if n < 3:
        return 1.0, n
    std = float(r.std(ddof=1))
    mean = float(r.mean())
    if not math.isfinite(std) or std <= 0.0:
        return (0.0 if mean > 0 else 1.0), n
    statistic = mean / (std / math.sqrt(n))
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
    """Apply FDR control to a population of candidate OOS return streams."""

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
