from __future__ import annotations

import pandas as pd


def _future_min_return(series: pd.Series, horizon: int) -> pd.Series:
    candidates = [series.shift(-step) / series - 1.0 for step in range(1, horizon + 1)]
    return pd.concat(candidates, axis=1).min(axis=1, skipna=False)


def make_panel_labels(panel: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 20), benchmark: pd.Series | None = None) -> pd.DataFrame:
    if not horizons or any(int(h) <= 0 for h in horizons):
        raise ValueError("horizons must be positive")
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names != ["timestamp", "symbol"]:
        raise ValueError("panel must be indexed by timestamp,symbol")
    close = panel["close"].astype(float)
    symbols = panel.index.get_level_values("symbol")
    out = pd.DataFrame(index=panel.index)
    for h in horizons:
        h = int(h)
        fwd = close.groupby(symbols).transform(lambda s: s.shift(-h) / s - 1.0)
        adverse = close.groupby(symbols, group_keys=False).apply(lambda s: _future_min_return(s, h)).reindex(panel.index)
        out[f"fwd_return_{h}"] = fwd
        out[f"adverse_excursion_{h}"] = adverse
        if benchmark is not None:
            bench = benchmark.astype(float).sort_index()
            benchmark_fwd = bench.shift(-h) / bench - 1.0
            ts = panel.index.get_level_values("timestamp")
            out[f"fwd_excess_return_{h}"] = fwd.to_numpy() - benchmark_fwd.reindex(ts).to_numpy()
    return out
