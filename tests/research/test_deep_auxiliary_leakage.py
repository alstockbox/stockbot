from types import SimpleNamespace

import pandas as pd

from stockbot.data.schemas import DataGrade, DatasetMetadata
from stockbot.research.deep_diagnostics import run_deep_research_diagnostics


HOLDOUT_START = pd.Timestamp("2025-03-01", tz="UTC")
AUXILIARY_STORE = object()


def _bars() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=80, freq="B", tz="UTC")
    rows = []
    for symbol in ("AAA", "BBB"):
        for i, dt in enumerate(dates):
            price = 100.0 + i
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": price,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price,
                    "volume": 1_000_000 + i,
                }
            )
    return pd.DataFrame(rows)


def _report():
    artifact = SimpleNamespace(
        model_name="ridge",
        model_params={"alpha": 1.0},
        seed=7,
        feature_names=("return_1", "aux__policy_rate"),
    )
    result = SimpleNamespace(
        artifact=artifact,
        predictions=pd.Series([1.0], dtype=float),
    )
    candidate = SimpleNamespace(
        experiment_id="aux-candidate-1",
        horizon=5,
        model_name="ridge",
        model_params={"alpha": 1.0},
        seed=7,
        factory_score=1.0,
        selection_score=2.0,
        oos_coverage=0.8,
        gate=SimpleNamespace(passed=True),
        result=result,
    )
    return SimpleNamespace(candidates=(candidate,), holdout_start=HOLDOUT_START)


def test_deep_auxiliary_replay_and_ablation_never_receive_blind_holdout(monkeypatch):
    seen = []

    def assert_research_only(bars):
        max_date = pd.to_datetime(bars["timestamp"], utc=True).max()
        seen.append(max_date)
        assert max_date < HOLDOUT_START

    def fake_factors(bars, *args, **kwargs):
        assert_research_only(bars)
        return pd.DataFrame({"market": [0.0] * 40})

    def fake_policy(bars, *args, **kwargs):
        assert_research_only(bars)
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

    def fake_liquidity(bars, *args, **kwargs):
        assert_research_only(bars)
        return SimpleNamespace(score=0.7)

    def fake_capacity(bars, *args, **kwargs):
        assert_research_only(bars)
        return SimpleNamespace(capacity_score=0.6)

    def assert_aux_replay(kwargs):
        assert kwargs["auxiliary_store"] is AUXILIARY_STORE
        assert tuple(kwargs["feature_columns"]) == ("return_1", "aux__policy_rate")

    def fake_windows(bars, *args, **kwargs):
        assert_research_only(bars)
        assert_aux_replay(kwargs)
        return SimpleNamespace(results=(), score=0.8, worst_score=0.5, positive_fraction=1.0)

    def fake_ablation(bars, *args, **kwargs):
        assert_research_only(bars)
        assert_aux_replay(kwargs)
        return SimpleNamespace(
            baseline_score=1.0,
            baseline_sharpe=1.0,
            baseline_cagr=0.1,
            results=(),
            recommended_drop_groups=(),
        )

    def fake_auxiliary_ablation(bars, *args, **kwargs):
        assert_research_only(bars)
        assert kwargs["auxiliary_store"] is AUXILIARY_STORE
        assert tuple(kwargs["feature_columns"]) == ("return_1", "aux__policy_rate")
        return SimpleNamespace(
            auxiliary_features=("policy_rate",),
            baseline_score=1.0,
            no_auxiliary_score=0.8,
            aggregate_score_impact=0.2,
            results=(),
            useful_features=("policy_rate",),
            harmful_features=(),
        )

    monkeypatch.setattr("stockbot.research.deep_diagnostics.build_internal_factor_returns", fake_factors)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_policy_arena", fake_policy)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_block_bootstrap_uncertainty", fake_bootstrap)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_factor_exposure", fake_factor_exposure)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.simulate_liquidity_aware_execution", fake_liquidity)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_capacity_curve", fake_capacity)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_training_window_robustness", fake_windows)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_feature_group_ablation", fake_ablation)
    monkeypatch.setattr("stockbot.research.deep_diagnostics.evaluate_auxiliary_feature_ablation", fake_auxiliary_ablation)

    diagnostics = run_deep_research_diagnostics(
        _bars(),
        DatasetMetadata("deep-aux-test", "synthetic", DataGrade.RESEARCH_GRADE),
        _report(),
        auxiliary_store=AUXILIARY_STORE,
        auxiliary_feature_names=("policy_rate",),
    )

    assert diagnostics.candidate_count == 1
    item = diagnostics.candidates["aux-candidate-1"]
    assert item.auxiliary_ablation is not None
    assert item.auxiliary_ablation.useful_features == ("policy_rate",)
    assert len(seen) == 7
    assert all(timestamp < HOLDOUT_START for timestamp in seen)
