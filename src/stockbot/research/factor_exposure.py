from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockbot.data.panel import build_panel
from stockbot.evaluation.metrics import performance_metrics


@dataclass(frozen=True)
class FactorExposureReport:
    factor_betas: dict[str, float]
    daily_alpha: float
    annualized_alpha: float
    r_squared: float
    residual_metrics: dict[str, float]
    residual_returns: pd.Series
    explained_fraction: float
    idiosyncratic_score: float
    observations: int


def _long_short_factor(asset_returns: pd.DataFrame, signal: pd.DataFrame, fraction: float = 0.30) -> pd.Series:
    values: list[float] = []
    for dt in asset_returns.index:
        ret = asset_returns.loc[dt]
        sig = signal.loc[dt]
        valid = pd.concat([ret.rename("ret"), sig.rename("sig")], axis=1).dropna()
        if len(valid) < 2:
            values.append(0.0)
            continue
        count = max(1, int(np.ceil(len(valid) * fraction)))
        ordered = valid.sort_values("sig")
        low = float(ordered.head(count)["ret"].mean())
        high = float(ordered.tail(count)["ret"].mean())
        values.append(high - low)
    return pd.Series(values, index=asset_returns.index, dtype=float)


def build_internal_factor_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """Build causal universe factors for post-hoc strategy exposure attribution.

    Sorting signals are lagged before current-day factor returns are calculated. These
    factors are diagnostics only: they help distinguish genuine residual edge from
    simple market, momentum, liquidity or volatility exposure.
    """

    panel = build_panel(bars)
    close = panel["close"].unstack("symbol").sort_index().astype(float)
    volume = panel["volume"].unstack("symbol").sort_index().astype(float)
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)

    market = returns.mean(axis=1).fillna(0.0)
    momentum_signal = close.pct_change(20).shift(1)
    dollar_volume = close * volume
    liquidity_signal = np.log1p(dollar_volume.rolling(20, min_periods=10).mean()).shift(1)
    volatility_signal = returns.rolling(20, min_periods=10).std(ddof=0).shift(1)

    factors = pd.DataFrame(
        {
            "market": market,
            "momentum": _long_short_factor(returns, momentum_signal),
            "liquidity": _long_short_factor(returns, liquidity_signal),
            "volatility": _long_short_factor(returns, volatility_signal),
        },
        index=close.index,
    )
    return factors.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def evaluate_factor_exposure(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> FactorExposureReport:
    """OLS-attribution of a strategy return stream against supplied factor returns."""

    strategy = pd.Series(strategy_returns, dtype=float).rename("strategy")
    aligned = pd.concat([strategy, factor_returns.astype(float)], axis=1, join="inner")
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna()
    if len(aligned) < max(30, len(factor_returns.columns) + 5):
        raise ValueError("insufficient aligned observations for factor attribution")

    y = aligned["strategy"].to_numpy(dtype=float)
    factor_names = list(factor_returns.columns)
    x_factors = aligned[factor_names].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(aligned)), x_factors])
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coefficients
    residual = y - fitted

    ss_total = float(np.sum((y - y.mean()) ** 2))
    ss_residual = float(np.sum(residual**2))
    r_squared = 0.0 if ss_total <= 0.0 else float(np.clip(1.0 - ss_residual / ss_total, 0.0, 1.0))
    explained_fraction = float(np.clip(r_squared, 0.0, 1.0))

    residual_series = pd.Series(residual, index=aligned.index, dtype=float, name="factor_residual_return")
    residual_metrics = performance_metrics(residual_series)
    residual_sharpe = float(residual_metrics.get("sharpe", 0.0))
    positive_residual = float(np.clip((residual_sharpe + 0.5) / 2.5, 0.0, 1.0))
    idiosyncratic_score = float(
        np.clip(0.65 * (1.0 - explained_fraction) + 0.35 * positive_residual, 0.0, 1.0)
    )

    daily_alpha = float(coefficients[0])
    annualized_alpha = float((1.0 + daily_alpha) ** 252 - 1.0) if daily_alpha > -1.0 else -1.0
    betas = {name: float(value) for name, value in zip(factor_names, coefficients[1:])}

    return FactorExposureReport(
        factor_betas=betas,
        daily_alpha=daily_alpha,
        annualized_alpha=annualized_alpha,
        r_squared=r_squared,
        residual_metrics={key: float(value) for key, value in residual_metrics.items()},
        residual_returns=residual_series,
        explained_fraction=explained_fraction,
        idiosyncratic_score=idiosyncratic_score,
        observations=len(aligned),
    )
