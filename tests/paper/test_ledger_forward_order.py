import pytest

from stockbot.paper.ledger import PaperObservation, PaperTradingLedger


def _row(strategy_id: str, timestamp: str, *, signal_timestamp: str | None = None) -> PaperObservation:
    return PaperObservation(
        strategy_id=strategy_id,
        timestamp=timestamp,
        net_return=0.001,
        turnover=0.2,
        cost_rate=0.0001,
        fill_rate=0.99,
        signal_count=2,
        signal_timestamp=signal_timestamp,
    )


def test_paper_observation_requires_timezone_aware_timestamps():
    with pytest.raises(ValueError, match="timezone-aware"):
        _row("strategy-a", "2026-09-09T16:00:00")

    with pytest.raises(ValueError, match="timezone-aware"):
        _row(
            "strategy-a",
            "2026-09-09T16:00:00+00:00",
            signal_timestamp="2026-09-08T16:00:00",
        )


def test_paper_observation_signal_must_precede_realization():
    with pytest.raises(ValueError, match="precede"):
        _row(
            "strategy-a",
            "2026-09-09T16:00:00+00:00",
            signal_timestamp="2026-09-09T16:00:00+00:00",
        )

    with pytest.raises(ValueError, match="precede"):
        _row(
            "strategy-a",
            "2026-09-09T16:00:00+00:00",
            signal_timestamp="2026-09-10T16:00:00+00:00",
        )


def test_paper_ledger_rejects_retroactive_backfill_per_strategy(tmp_path):
    ledger = PaperTradingLedger(tmp_path / "paper.jsonl")
    latest = _row("strategy-a", "2026-09-09T16:00:00+00:00")
    ledger.append(latest)

    # A different strategy has an independent forward timeline and may legitimately
    # begin with an earlier observation timestamp.
    other = _row("strategy-b", "2026-09-08T16:00:00+00:00")
    ledger.append(other)

    retroactive = _row("strategy-a", "2026-09-08T16:00:00+00:00")
    with pytest.raises(ValueError, match="strictly increasing"):
        ledger.append(retroactive)

    assert ledger.records(strategy_id="strategy-a") == [latest]
    assert ledger.records(strategy_id="strategy-b") == [other]
