import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import stockbot.research.quarantine_audit as quarantine_audit_module
from stockbot.research.holdout import HoldoutConfig
from stockbot.research.quarantine import QuarantineConfig
from stockbot.research.quarantine_audit import freeze_strategy_spec, run_single_quarantine_audit


def _bars() -> pd.DataFrame:
    rng = np.random.default_rng(20260908)
    dates = pd.date_range("2024-01-02", periods=360, freq="B", tz="UTC")
    rows = []
    market = rng.normal(0.0002, 0.004, len(dates))
    for j, symbol in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")):
        local = market + 0.0005 * np.sin(np.arange(len(dates)) / (8.0 + j)) + rng.normal(0.0, 0.003, len(dates))
        close = 100.0 * np.exp(np.cumsum(local))
        volume = rng.integers(700_000 + 80_000 * j, 1_400_000 + 120_000 * j, len(dates))
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": float(close[i]),
                    "high": float(close[i] * 1.006),
                    "low": float(close[i] * 0.994),
                    "close": float(close[i]),
                    "volume": int(volume[i]),
                }
            )
    return pd.DataFrame(rows)


def _candidate(experiment_id: str, alpha: float):
    return SimpleNamespace(
        experiment_id=experiment_id,
        horizon=5,
        model_name="ridge",
        model_params={"alpha": alpha},
        seed=7,
    )


def _quarantine() -> QuarantineConfig:
    return QuarantineConfig(
        start="2025-03-03",
        min_development_periods=250,
        min_quarantine_periods=40,
    )


def _relaxed_holdout() -> HoldoutConfig:
    return HoldoutConfig(
        fraction=0.15,
        min_holdout_periods=40,
        min_research_periods=126,
        min_prediction_coverage=0.20,
        max_drawdown=1.0,
        min_stress_score=0.0,
        min_score=-1_000_000.0,
    )


def test_quarantine_audit_allows_only_one_frozen_strategy_per_cycle(tmp_path):
    bars = _bars()
    quarantine = _quarantine()
    relaxed = _relaxed_holdout()
    first_spec = freeze_strategy_spec(_candidate("candidate-a", 1.0), top_fraction=0.25, weighting="equal")
    ledger = tmp_path / "audit-ledger.json"

    result = run_single_quarantine_audit(
        bars,
        quarantine,
        first_spec,
        ledger_path=ledger,
        holdout_config=relaxed,
    )
    assert result.record.strategy_id == first_spec.strategy_id
    assert result.holdout_report.periods >= 40
    assert ledger.exists()

    try:
        run_single_quarantine_audit(
            bars,
            quarantine,
            first_spec,
            ledger_path=ledger,
            holdout_config=relaxed,
        )
    except ValueError as exc:
        assert "already been audited" in str(exc)
    else:
        raise AssertionError("expected repeat audit to fail")

    second_spec = freeze_strategy_spec(_candidate("candidate-b", 10.0), top_fraction=0.25, weighting="equal")
    try:
        run_single_quarantine_audit(
            bars,
            quarantine,
            second_spec,
            ledger_path=ledger,
            holdout_config=relaxed,
        )
    except ValueError as exc:
        assert "another strategy" in str(exc)
    else:
        raise AssertionError("expected second strategy audit to fail")


def test_quarantine_audit_rejects_tampered_persisted_result(tmp_path):
    bars = _bars()
    quarantine = _quarantine()
    spec = freeze_strategy_spec(_candidate("candidate-a", 1.0), top_fraction=0.25, weighting="equal")
    ledger = tmp_path / "audit-ledger.json"

    run_single_quarantine_audit(
        bars,
        quarantine,
        spec,
        ledger_path=ledger,
        holdout_config=_relaxed_holdout(),
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload[0]["passed"] = not bool(payload[0]["passed"])
    ledger.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="audit integrity"):
        run_single_quarantine_audit(
            bars,
            quarantine,
            spec,
            ledger_path=ledger,
            holdout_config=_relaxed_holdout(),
        )


def test_verified_quarantine_audit_lookup_requires_matching_cycle_and_strategy(tmp_path):
    bars = _bars()
    quarantine = _quarantine()
    spec = freeze_strategy_spec(_candidate("candidate-a", 1.0), top_fraction=0.25, weighting="equal")
    ledger = tmp_path / "audit-ledger.json"

    result = run_single_quarantine_audit(
        bars,
        quarantine,
        spec,
        ledger_path=ledger,
        holdout_config=_relaxed_holdout(),
    )

    record = quarantine_audit_module.load_verified_quarantine_audit_record(
        ledger,
        quarantine_start=result.manifest.start,
        strategy_id=spec.strategy_id,
    )
    assert record.audit_id == result.record.audit_id
    assert record.strategy_id == spec.strategy_id

    with pytest.raises(ValueError, match="strategy"):
        quarantine_audit_module.load_verified_quarantine_audit_record(
            ledger,
            quarantine_start=result.manifest.start,
            strategy_id="different-strategy",
        )
    with pytest.raises(ValueError, match="not found"):
        quarantine_audit_module.load_verified_quarantine_audit_record(
            ledger,
            quarantine_start="2030-01-01",
            strategy_id=spec.strategy_id,
        )


def test_verified_quarantine_audit_rejects_tampered_audited_at(tmp_path):
    bars = _bars()
    quarantine = _quarantine()
    spec = freeze_strategy_spec(_candidate("candidate-a", 1.0), top_fraction=0.25, weighting="equal")
    ledger = tmp_path / "audit-ledger.json"

    result = run_single_quarantine_audit(
        bars,
        quarantine,
        spec,
        ledger_path=ledger,
        holdout_config=_relaxed_holdout(),
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload[0]["audited_at"] = "2035-01-01T00:00:00+00:00"
    ledger.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="audit integrity"):
        quarantine_audit_module.load_verified_quarantine_audit_record(
            ledger,
            quarantine_start=result.manifest.start,
            strategy_id=spec.strategy_id,
        )


def test_verified_quarantine_audit_rejects_duplicate_cycle_records(tmp_path):
    bars = _bars()
    quarantine = _quarantine()
    spec = freeze_strategy_spec(_candidate("candidate-a", 1.0), top_fraction=0.25, weighting="equal")
    ledger = tmp_path / "audit-ledger.json"

    result = run_single_quarantine_audit(
        bars,
        quarantine,
        spec,
        ledger_path=ledger,
        holdout_config=_relaxed_holdout(),
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload.append(dict(payload[0]))
    ledger.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="multiple.*audit"):
        quarantine_audit_module.load_verified_quarantine_audit_record(
            ledger,
            quarantine_start=result.manifest.start,
            strategy_id=spec.strategy_id,
        )


def test_quarantine_audit_persists_and_verifies_research_cycle(tmp_path):
    bars = _bars()
    quarantine = _quarantine()
    spec = freeze_strategy_spec(_candidate("candidate-a", 1.0), top_fraction=0.25, weighting="equal")
    ledger = tmp_path / "audit-ledger.json"

    result = run_single_quarantine_audit(
        bars,
        quarantine,
        spec,
        ledger_path=ledger,
        holdout_config=_relaxed_holdout(),
        research_cycle_id="cycle-a",
    )
    assert result.record.research_cycle_id == "cycle-a"

    record = quarantine_audit_module.load_verified_quarantine_audit_record(
        ledger,
        quarantine_start=result.manifest.start,
        strategy_id=spec.strategy_id,
        research_cycle_id="cycle-a",
    )
    assert record.audit_id == result.record.audit_id
    assert record.research_cycle_id == "cycle-a"

    with pytest.raises(ValueError, match="research cycle"):
        quarantine_audit_module.load_verified_quarantine_audit_record(
            ledger,
            quarantine_start=result.manifest.start,
            strategy_id=spec.strategy_id,
            research_cycle_id="different-cycle",
        )
