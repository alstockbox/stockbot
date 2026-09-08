from types import SimpleNamespace

import pandas as pd

from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.research.deep_diagnostics import run_deep_research_diagnostics


class _Gate:
    passed = True


class _Artifact:
    model_name = "ridge"
    model_params = {"alpha": 1.0}
    seed = 7


class _Result:
    artifact = _Artifact()


class _Candidate:
    experiment_id = "candidate-1"
    horizon = 5
    model_name = "ridge"
    model_params = {"alpha": 1.0}
    seed = 7
    factory_score = 1.0
    selection_score = 2.0
    gate = _Gate()
    result = _Result()


class _Report:
    candidates = (_Candidate(),)
    holdout_start = pd.Timestamp("2025-03-01", tz="UTC")


def _bars() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=80, freq="B", tz="UTC")
    rows = []
    for symbol in ("AAA", "BBB"):
        for i, dt in enumerate(dates):
            price = 100.0 + i
            rows.append({
                "symbol": symbol,
                "timestamp": dt,
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": 1_000_000,
            })
    return pd.DataFrame(rows)


def test_deep_diagnostics_never_pass_blind_holdout_rows_to_subresearch(monkeypatch):
    seen_max_dates = []

    def _assert_research_only(bars, *args, **kwargs):
        max_date = pd.to_datetime(bars["timestamp"], utc=True).max()
        seen_max_dates.append(max_date)
        assert max_date < _Report.holdout_start

    def fake_policy(bars, *args, **kwargs):
        _assert_research_only(bars)
        return SimpleNamespace(best=SimpleNamespace(score=1.0), baseline=None)

    def fake_windows(bars, *args, **kwargs):
        _assert_research_only(bars)
        return SimpleNamespace(results=(), score=0.8, worst_score=0.5, positive_fraction=1.0)

    def fake_ablation(bars, *args, **kwargs):
        _assert_research_only(bars)
        return SimpleNamespace(
            baseline_score=1.0,
            baseline_sharpe=1.0,
            baseline_cagr=0.1,
            results=(),
            recommended_drop_groups=(),
        )

    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_policy_arena", fake_policy)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_training_window_robustness", fake_windows)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_feature_group_ablation", fake_ablation)

    metadata = DatasetMetadata("deep-test", "synthetic", DataGrade.RESEARCH_GRADE)
    diagnostics = run_deep_research_diagnostics(_bars(), metadata, _Report())

    assert diagnostics.candidate_count == 1
    assert len(seen_max_dates) == 3
