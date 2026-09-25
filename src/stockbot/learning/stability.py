from __future__ import annotations

import math

import numpy as np
import pandas as pd


def monthly_compounded_returns(returns: pd.Series) -> pd.Series:
    clean = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if not isinstance(clean.index, pd.DatetimeIndex):
        raise ValueError("monthly stability requires a DatetimeIndex")
    if clean.empty:
        return pd.Series(dtype=float)
    periods = clean.index.to_period("M")
    monthly = (1.0 + clean).groupby(periods).prod() - 1.0
    return monthly.astype(float)


def stability_metrics(returns: pd.Series) -> dict[str, float]:
    monthly = monthly_compounded_returns(returns)
    if monthly.empty:
        return {
            "negative_month_rate": 0.0,
            "monthly_observations": 0.0,
            "worst_month": 0.0,
            "monthly_return_volatility": 0.0,
            "average_loss_month": 0.0,
        }

    losses = monthly[monthly < 0.0]
    volatility = float(monthly.std(ddof=0)) if len(monthly) > 1 else 0.0
    avg_loss = float(-losses.mean()) if len(losses) else 0.0
    values = {
        "negative_month_rate": float((monthly < 0.0).mean()),
        "monthly_observations": float(len(monthly)),
        "worst_month": float(monthly.min()),
        "monthly_return_volatility": volatility,
        "average_loss_month": avg_loss,
    }
    return {key: value if math.isfinite(value) else 0.0 for key, value in values.items()}
