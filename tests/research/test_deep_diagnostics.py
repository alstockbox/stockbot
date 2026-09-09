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
    predictions = pd.Series([1.0])


class _Candidate:
    experiment_id = "candidate-1"
    horizon = 5
    model_name = "ridge"
    model_params = {"alpha": 1.0}
    seed = 7
    factory_score = 1.0
    selection_score = 2.0
    oos_coverage = 0.8
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

    def fake_factors(bars, *args, **kwargs):
        _assert_research_only(bars)
        return pd.DataFrame({"market": [0.0] * 40})

    def fake_policy(bars, *args, **kwargs):
        _assert_research_only(bars)
        policy = SimpleNamespace(top_fraction=0.30, weighting="equal")
        best = SimpleNamespace(
            score=1.0,
            policy=policy,
            net_returns=pd.Series([0.001] * 40, dtype=float),
        )
        return SimpleNamespace(best=best, baseline=None)

    def fake_bootstrap(*args, **kwargs):
        return SimpleNamespace(confidence_score=0.9)

    def fake_factor_exposure(*args, **kwargs):
        return SimpleNamespace(idiosyncratic_score=0.8)

    def fake_neutralization(bars, *args, **kwargs):
        _assert_research_only(bars)
        return SimpleNamespace(neutralized_score=0.9)

    def fake_sector_cap(bars, *args, **kwargs):
        _assert_research_only(bars)
        return SimpleNamespace(constrained_score=1.1, max_sector_weight=0.35)

    def fake_liquidity(bars, *args, **kwargs):
        _assert_research_only(bars)
        return SimpleNamespace(score=0.7)

    def fake_capacity(bars, *args, **kwargs):
        _assert_research_only(bars)
        return SimpleNamespace(capacity_score=0.6)

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

    monkeypatch.setattr("stockbot.research.deep_diagnostics.build_internal_factor_returns", fake_factors)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_policy_arena", fake_policy)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_block_bootstrap_uncertainty", fake_bootstrap)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_factor_exposure", fake_factor_exposure)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_sector_factor_neutralization", fake_neutralization)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_sector_cap_challenger", fake_sector_cap)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.simulate_liquidity_aware_execution", fake_liquidity)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_capacity_curve", fake_capacity)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_training_window_robustness", fake_windows)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_feature_group_ablation", fake_ablation)

    metadata = DatasetMetadata("deep-test", "synthetic", DataGrade.RESEARCH_GRADE)
    diagnostics = run_deep_research_diagnostics(
        _bars(),
        metadata,
        _Report(),
        point_in_time_universe=object(),
    )

    assert diagnostics.candidate_count == 1
    assert diagnostics.candidates["candidate-1"].sector_cap.constrained_score == 1.1
    assert len(seen_max_dates) == 8
