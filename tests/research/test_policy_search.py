import numpy as np
import pandas as pd

from stockbot.arena.experiments import ModelExperimentResult
from stockbot.data.panel import build_panel
from stockbot.ml.artifacts import ExperimentArtifact
from stockbot.research.policy_search import evaluate_policy_arena


def _bars() -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=90, freq="B", tz="UTC")
    rows = []
    for j, symbol in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        returns = np.full(len(dates), 0.0002 + j * 0.0002)
        close = 100.0 * np.cumprod(1.0 + returns)
        for i, dt in enumerate(dates):
            rows.append({
                "symbol": symbol,
                "timestamp": dt,
                "open": close[i],
                "high": close[i] * 1.005,
                "low": close[i] * 0.995,
                "close": close[i],
                "volume": 1_000_000 + j * 100_000,
            })
    return pd.DataFrame(rows)


def _result(bars: pd.DataFrame) -> ModelExperimentResult:
    panel = build_panel(bars)
    symbols = panel.index.get_level_values("symbol")
    values = pd.Series(symbols.map({"AAA": 0.1, "BBB": 0.2, "CCC": 0.5, "DDD": 1.0}), index=panel.index, dtype=float)
    artifact = ExperimentArtifact(
        dataset_fingerprint="policy-test",
        feature_names=("x",),
        label_name="target",
        model_name="ridge",
        model_params={"alpha": 1.0},
        seed=7,
        fold_count=3,
        oos_coverage=1.0,
        metrics={},
    )
    return ModelExperimentResult(
        name="ridge",
        score=1.0,
        metrics={},
        robustness=0.8,
        oos_coverage=1.0,
        artifact=artifact,
        predictions=values,
    )


def test_policy_arena_reuses_predictions_across_selectivity_and_weighting_modes():
    bars = _bars()
    report = evaluate_policy_arena(
        bars,
        _result(bars),
        top_fractions=(0.25, 0.50),
        weightings=("equal", "conviction"),
    )

    assert len(report.results) == 4
    assert report.best in report.results
    assert all(np.isfinite(result.score) for result in report.results)
    assert all(0.0 <= result.stress_report.score <= 1.0 for result in report.results)
    policies = {(result.policy.top_fraction, result.policy.weighting) for result in report.results}
    assert policies == {(0.25, "equal"), (0.25, "conviction"), (0.50, "equal"), (0.50, "conviction")}
