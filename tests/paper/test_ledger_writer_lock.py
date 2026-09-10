import fcntl

import pytest

from stockbot.paper.ledger import PaperObservation, PaperTradingLedger


def _row(day: int) -> PaperObservation:
    return PaperObservation(
        strategy_id="strategy-writer-lock",
        timestamp=f"2026-09-{day:02d}T16:00:00+00:00",
        net_return=0.001 * day,
        turnover=0.2,
        cost_rate=0.0001,
        fill_rate=0.99,
        signal_count=2,
    )


def test_paper_ledger_append_fails_closed_when_writer_lock_is_held(tmp_path):
    path = tmp_path / "paper.jsonl"
    ledger = PaperTradingLedger(path)
    first = _row(1)
    second = _row(2)
    ledger.append(first)
    before = path.read_bytes()

    lock_path = path.with_name(path.name + ".lock")
    holder = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(ValueError, match="already being written"):
            ledger.append(second)
        assert path.read_bytes() == before
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()

    # Verified readers coordinate with the same lock so they cannot inspect the
    # transient ledger/checkpoint interval of a real append commit.
    assert ledger.records() == [first]


def test_paper_ledger_writer_lock_releases_after_successful_append(tmp_path):
    path = tmp_path / "paper.jsonl"
    ledger = PaperTradingLedger(path)
    ledger.append(_row(1))
    ledger.append(_row(2))

    assert ledger.records() == [_row(1), _row(2)]

    lock_path = path.with_name(path.name + ".lock")
    probe = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
        probe.close()
