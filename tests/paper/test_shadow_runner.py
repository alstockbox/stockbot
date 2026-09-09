from pathlib import Path

import pandas as pd

import stockbot.paper.runner as shadow_runner
from stockbot.paper.ledger import PaperTradingLedger
from stockbot.paper.runner import (
    freeze_shadow_strategy,
    load_frozen_shadow_artifact,
    run_shadow_step,
)
from stockbot.research.liquidity_execution import LiquidityExecutionConfig
from stockbot.research.quarantine_audit import FrozenStrategySpec
from stockbot.risk.engine import RiskConfig


def _bars(periods: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=periods, freq="B", tz="UTC")
    rows = []
    for offset, symbol in enumerate(("AAA", "BBB", "CCC")):
        base = 90.0 + 15.0 * offset
        for i, dt in enumerate(dates):
            # Deterministic but non-identical upward paths keep cross-sectional ranks
            # and short momentum features informative without any random fixture state.
            close = base * (1.0 + 0.0015 * i + 0.00015 * offset * i)
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": close * 0.998,
                    "high": close * 1.004,
                    "low": close * 0.996,
                    "close": close,
                    "volume": 1_000_000.0 + 50_000.0 * offset + 2_000.0 * i,
                }
            )
    return pd.DataFrame(rows).sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)


def _spec() -> FrozenStrategySpec:
    return FrozenStrategySpec(
        strategy_id="strategy-shadow",
        experiment_id="exp-shadow",
        horizon=1,
        model_name="ridge",
        model_params={"alpha": 1.0},
        seed=7,
        top_fraction=0.50,
        weighting="equal",
    )


def _freeze(tmp_path: Path, bars: pd.DataFrame):
    return freeze_shadow_strategy(
        bars,
        _spec(),
        feature_names=(
            "return_1",
            "momentum_5",
            "realized_vol_5",
            "log_dollar_volume",
            "return_1_rank",
        ),
        research_cycle_id="cycle-paper-forward",
        source_dataset_fingerprint="research-snapshot-fingerprint",
        output_dir=tmp_path / "artifact",
        execution_config=LiquidityExecutionConfig(
            capital=100_000.0,
            commission_bps=1.0,
            spread_bps=4.0,
            impact_bps_at_one_pct_adv=10.0,
            max_participation=0.05,
            adv_window=5,
        ),
        risk_config=RiskConfig(
            max_position_weight=0.60,
            max_drawdown=0.20,
            max_daily_loss=0.10,
            max_data_age_seconds=86_400.0,
            abnormal_volatility=2.0,
        ),
    )


def test_freeze_and_two_shadow_steps_create_forward_observation_with_provenance(tmp_path):
    all_bars = _bars()
    freeze_dates = sorted(all_bars["timestamp"].unique())[:75]
    freeze_bars = all_bars[all_bars["timestamp"].isin(freeze_dates)].copy()
    next_date = sorted(all_bars["timestamp"].unique())[75]
    next_bars = all_bars[all_bars["timestamp"] <= next_date].copy()

    manifest = _freeze(tmp_path, freeze_bars)
    assert manifest.strategy_id == "strategy-shadow"
    assert manifest.research_cycle_id == "cycle-paper-forward"
    assert manifest.source_dataset_fingerprint == "research-snapshot-fingerprint"
    assert manifest.model_sha256
    assert manifest.feature_names == (
        "return_1",
        "momentum_5",
        "realized_vol_5",
        "log_dollar_volume",
        "return_1_rank",
    )
    assert not manifest.broker_execution_available

    state_path = tmp_path / "paper" / "state.json"
    ledger_path = tmp_path / "paper" / "observations.jsonl"

    first = run_shadow_step(
        freeze_bars,
        artifact_dir=tmp_path / "artifact",
        state_path=state_path,
        ledger_path=ledger_path,
        snapshot_fingerprint="paper-snapshot-001",
        data_age_seconds=0.0,
    )
    assert first.observation is None
    assert first.pending_signal_timestamp == pd.Timestamp(freeze_dates[-1]).isoformat()
    assert not first.broker_execution_available
    assert PaperTradingLedger(ledger_path).records(strategy_id="strategy-shadow") == []

    second = run_shadow_step(
        next_bars,
        artifact_dir=tmp_path / "artifact",
        state_path=state_path,
        ledger_path=ledger_path,
        snapshot_fingerprint="paper-snapshot-002",
        data_age_seconds=0.0,
    )
    assert second.observation is not None
    observation = second.observation
    assert observation.strategy_id == "strategy-shadow"
    assert observation.research_cycle_id == "cycle-paper-forward"
    assert observation.model_artifact_id == manifest.artifact_id
    assert observation.signal_timestamp == pd.Timestamp(freeze_dates[-1]).isoformat()
    assert observation.signal_snapshot_fingerprint == "paper-snapshot-001"
    assert observation.realization_snapshot_fingerprint == "paper-snapshot-002"
    assert observation.timestamp == pd.Timestamp(next_date).isoformat()
    assert 0.0 <= observation.fill_rate <= 1.0
    assert observation.cost_rate >= 0.0
    assert not second.broker_execution_available

    records = PaperTradingLedger(ledger_path).records(strategy_id="strategy-shadow")
    assert records == [observation]


def test_shadow_step_same_snapshot_is_idempotent_noop(tmp_path):
    all_bars = _bars()
    freeze_dates = sorted(all_bars["timestamp"].unique())[:75]
    freeze_bars = all_bars[all_bars["timestamp"].isin(freeze_dates)].copy()
    _freeze(tmp_path, freeze_bars)

    kwargs = dict(
        artifact_dir=tmp_path / "artifact",
        state_path=tmp_path / "paper" / "state.json",
        ledger_path=tmp_path / "paper" / "observations.jsonl",
        snapshot_fingerprint="paper-snapshot-001",
        data_age_seconds=0.0,
    )
    first = run_shadow_step(freeze_bars, **kwargs)
    replay = run_shadow_step(freeze_bars, **kwargs)

    assert replay.processed_timestamp == first.processed_timestamp
    assert replay.pending_signal_timestamp == first.pending_signal_timestamp
    assert replay.target_weights == first.target_weights
    assert replay.observation is None
    assert PaperTradingLedger(kwargs["ledger_path"]).records(strategy_id="strategy-shadow") == []


def test_shadow_step_rejects_same_timestamp_from_different_snapshot(tmp_path):
    all_bars = _bars()
    freeze_dates = sorted(all_bars["timestamp"].unique())[:75]
    freeze_bars = all_bars[all_bars["timestamp"].isin(freeze_dates)].copy()
    _freeze(tmp_path, freeze_bars)

    base_kwargs = dict(
        artifact_dir=tmp_path / "artifact",
        state_path=tmp_path / "paper" / "state.json",
        ledger_path=tmp_path / "paper" / "observations.jsonl",
        data_age_seconds=0.0,
    )
    run_shadow_step(freeze_bars, snapshot_fingerprint="paper-snapshot-001", **base_kwargs)

    try:
        run_shadow_step(freeze_bars, snapshot_fingerprint="paper-snapshot-rewritten", **base_kwargs)
    except ValueError as exc:
        assert "snapshot" in str(exc).lower()
    else:
        raise AssertionError("same market timestamp from a different snapshot must fail closed")


def test_shadow_step_recovers_if_ledger_was_written_before_state_crash(tmp_path, monkeypatch):
    all_bars = _bars()
    freeze_dates = sorted(all_bars["timestamp"].unique())[:75]
    freeze_bars = all_bars[all_bars["timestamp"].isin(freeze_dates)].copy()
    next_date = sorted(all_bars["timestamp"].unique())[75]
    next_bars = all_bars[all_bars["timestamp"] <= next_date].copy()
    _freeze(tmp_path, freeze_bars)

    state_path = tmp_path / "paper" / "state.json"
    ledger_path = tmp_path / "paper" / "observations.jsonl"
    run_shadow_step(
        freeze_bars,
        artifact_dir=tmp_path / "artifact",
        state_path=state_path,
        ledger_path=ledger_path,
        snapshot_fingerprint="paper-snapshot-001",
    )

    original_save_state = shadow_runner._save_state

    def crash_after_ledger_write(path, state):
        raise RuntimeError("simulated crash before state commit")

    monkeypatch.setattr(shadow_runner, "_save_state", crash_after_ledger_write)
    try:
        run_shadow_step(
            next_bars,
            artifact_dir=tmp_path / "artifact",
            state_path=state_path,
            ledger_path=ledger_path,
            snapshot_fingerprint="paper-snapshot-002",
        )
    except RuntimeError as exc:
        assert "simulated crash" in str(exc)
    else:
        raise AssertionError("simulated state-write crash must propagate")

    rows_after_crash = PaperTradingLedger(ledger_path).records(strategy_id="strategy-shadow")
    assert len(rows_after_crash) == 1
    assert rows_after_crash[0].timestamp == pd.Timestamp(next_date).isoformat()

    monkeypatch.setattr(shadow_runner, "_save_state", original_save_state)
    recovered = run_shadow_step(
        next_bars,
        artifact_dir=tmp_path / "artifact",
        state_path=state_path,
        ledger_path=ledger_path,
        snapshot_fingerprint="paper-snapshot-002",
    )

    rows_after_recovery = PaperTradingLedger(ledger_path).records(strategy_id="strategy-shadow")
    assert len(rows_after_recovery) == 1
    assert recovered.observation == rows_after_recovery[0]
    assert recovered.processed_timestamp == pd.Timestamp(next_date).isoformat()


def test_loading_frozen_shadow_artifact_rejects_model_tampering(tmp_path):
    all_bars = _bars()
    freeze_dates = sorted(all_bars["timestamp"].unique())[:75]
    freeze_bars = all_bars[all_bars["timestamp"].isin(freeze_dates)].copy()
    manifest = _freeze(tmp_path, freeze_bars)

    model_path = tmp_path / "artifact" / manifest.model_filename
    with model_path.open("ab") as handle:
        handle.write(b"tampered")

    try:
        load_frozen_shadow_artifact(tmp_path / "artifact")
    except ValueError as exc:
        assert "model hash" in str(exc)
    else:
        raise AssertionError("tampered frozen model bytes must fail closed")
