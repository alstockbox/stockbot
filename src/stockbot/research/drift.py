from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from stockbot.evaluation.metrics import performance_metrics


@dataclass(frozen=True)
class DriftReport:
    reference_metrics: dict[str, float]
    recent_metrics: dict[str, float]
    mean_shift_z: float
    volatility_ratio: float
    sharpe_delta: float
    score: float
    degraded: bool
    reasons: tuple[str, ...]


def evaluate_return_drift(
    returns: pd.Series,
    *,
    recent_fraction: float = 0.30,
    min_reference: int = 40,
    min_recent: int = 20,
    max_mean_shift_z: float = 1.5,
    max_volatility_ratio: float = 1.8,
    max_sharpe_drop: float = 1.0,
) -> DriftReport:
    """Detect whether recent OOS behavior has materially degraded versus its own history."""

    if not 0.10 <= recent_fraction <= 0.50:
        raise ValueError("recent_fraction must be in [0.10,0.50]")
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(r)
    recent_n = max(min_recent, int(math.ceil(n * recent_fraction)))
    if n - recent_n < min_reference or recent_n > n:
        empty = {"cagr": 0.0, "sharpe": 0.0, "volatility": 0.0, "max_drawdown": 0.0}
        return DriftReport(empty, empty, 0.0, 1.0, 0.0, 0.0, True, ("insufficient_history",))

    reference = r.iloc[: n - recent_n]
    recent = r.iloc[n - recent_n :]
    reference_metrics = performance_metrics(reference)
    recent_metrics = performance_metrics(recent)

    ref_mean = float(reference.mean())
    recent_mean = float(recent.mean())
    ref_std = float(reference.std(ddof=1))
    if not math.isfinite(ref_std) or ref_std <= 0.0:
        mean_shift_z = 0.0 if math.isclose(ref_mean, recent_mean) else float("inf")
    else:
        mean_shift_z = abs(recent_mean - ref_mean) / (ref_std / math.sqrt(max(1, len(recent))))

    ref_vol = max(1e-12, float(reference_metrics.get("volatility", 0.0)))
    recent_vol = max(0.0, float(recent_metrics.get("volatility", 0.0)))
    volatility_ratio = recent_vol / ref_vol
    sharpe_delta = float(recent_metrics.get("sharpe", 0.0)) - float(reference_metrics.get("sharpe", 0.0))

    reasons: list[str] = []
    if mean_shift_z > max_mean_shift_z and recent_mean < ref_mean:
        reasons.append("negative_mean_shift")
    if volatility_ratio > max_volatility_ratio:
        reasons.append("volatility_spike")
    if sharpe_delta < -max_sharpe_drop:
        reasons.append("sharpe_degradation")
    if float(recent_metrics.get("max_drawdown", 0.0)) > float(reference_metrics.get("max_drawdown", 0.0)) + 0.10:
        reasons.append("drawdown_degradation")

    mean_component = float(np.clip(1.0 - max(0.0, mean_shift_z - 0.5) / 3.0, 0.0, 1.0))
    vol_component = float(np.clip(1.0 - max(0.0, volatility_ratio - 1.0) / 2.0, 0.0, 1.0))
    sharpe_component = float(np.clip(1.0 + min(0.0, sharpe_delta) / 2.0, 0.0, 1.0))
    score = float(np.clip(0.40 * mean_component + 0.25 * vol_component + 0.35 * sharpe_component, 0.0, 1.0))

    return DriftReport(
        reference_metrics={key: float(value) for key, value in reference_metrics.items()},
        recent_metrics={key: float(value) for key, value in recent_metrics.items()},
        mean_shift_z=float(mean_shift_z),
        volatility_ratio=float(volatility_ratio),
        sharpe_delta=float(sharpe_delta),
        score=score,
        degraded=bool(reasons),
        reasons=tuple(reasons),
    )
