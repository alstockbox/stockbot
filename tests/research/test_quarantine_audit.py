from types import SimpleNamespace

import numpy as np
import pandas as pd

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
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": float(close[i]),
                    "high": float(close[i] * 1.006),
                    "low": float(close[i] * 0.994),
                    "close": float(close[i]),
                    "volume": int(800_000 + 100_000 * j),
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


def test_quarantine_audit_allows_only_one_frozen_strategy_per_cycle(tmp_path):
    bars = _bars()
    quarantine = QuarantineConfig(
        start="2025-03-03",
        min_development_periods=250,
        min_quarantine_periods=40,
    )
    relaxed = HoldoutConfig(
        fraction=0.15,
        min_holdout_periods=40,
        min_research_periods=126,
        min_prediction_coverage=0.20,
        max_drawdown=1.0,
        min_stress_score=0.0,
        min_score=-1_000_000.0,
    )
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
