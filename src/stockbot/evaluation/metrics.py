from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _finite(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0


def _cagr(returns: pd.Series, periods_per_year: int) -> float:
    if len(returns) == 0:
        return 0.0
    wealth = float((1.0 + returns).prod())
    if wealth <= 0:
        return -1.0
    years = len(returns) / periods_per_year
    if years <= 0:
        return 0.0
    return wealth ** (1.0 / years) - 1.0


def performance_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cagr = _cagr(r, periods_per_year)
    vol = float(r.std(ddof=0) * math.sqrt(periods_per_year))
    mean_ann = float(r.mean() * periods_per_year)
    sharpe = mean_ann / vol if vol > 1e-12 else 0.0

    downside = r[r < 0]
    downside_dev = float(downside.std(ddof=0) * math.sqrt(periods_per_year)) if len(downside) else 0.0
    sortino = mean_ann / downside_dev if downside_dev > 1e-12 else 0.0

    wealth = (1.0 + r).cumprod()
    rolling_peak = wealth.cummax()
    dd = 1.0 - wealth / rolling_peak.replace(0, np.nan)
    max_drawdown = float(dd.max()) if len(dd) else 0.0
    max_drawdown = max(0.0, _finite(max_drawdown))
    calmar = cagr / max_drawdown if max_drawdown > 1e-12 else 0.0

    q = float(r.quantile(0.05)) if len(r) else 0.0
    tail = r[r <= q]
    cvar = max(0.0, -float(tail.mean())) if len(tail) else 0.0

    wins = r[r > 0]
    losses = r[r < 0]
    hit_rate = float((r > 0).mean()) if len(r) else 0.0
    profit_factor = float(wins.sum() / abs(losses.sum())) if abs(float(losses.sum())) > 1e-12 else 0.0

    excess = 0.0
    beta = 0.0
    alpha = 0.0
    if benchmark_returns is not None:
        b = pd.Series(benchmark_returns, dtype=float).reindex(r.index).fillna(0.0)
        excess = cagr - _cagr(b, periods_per_year)
        bvar = float(b.var(ddof=0))
        if bvar > 1e-12:
            beta = float(np.cov(r, b, ddof=0)[0, 1] / bvar)
        alpha = mean_ann - beta * float(b.mean() * periods_per_year)

    return {
        "cagr": _finite(cagr),
        "annual_return": _finite(mean_ann),
        "volatility": _finite(vol),
        "sharpe": _finite(sharpe),
        "sortino": _finite(sortino),
        "max_drawdown": _finite(max_drawdown),
        "calmar": _finite(calmar),
        "cvar_95": _finite(cvar),
        "hit_rate": _finite(hit_rate),
        "profit_factor": _finite(profit_factor),
        "excess_return": _finite(excess),
        "beta": _finite(beta),
        "alpha": _finite(alpha),
    }
