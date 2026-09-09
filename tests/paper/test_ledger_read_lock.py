import threading

from stockbot.paper.ledger import PaperObservation, PaperTradingLedger
from stockbot.paper.locking import (
    exclusive_paper_ledger_write_lock,
    paper_ledger_write_lock_path,
)


def _row(day: int) -> PaperObservation:
    return PaperObservation(
        strategy_id="strategy-read-lock",
        timestamp=f"2026-09-{day:02d}T16:00:00+00:00",
        net_return=0.001,
        turnover=0.2,
        cost_rate=0.0001,
        fill_rate=0.99,
        signal_count=3,
    )


def test_verified_ledger_read_waits_for_inflight_writer_commit(tmp_path):
    path = tmp_path / "paper.jsonl"
    ledger = PaperTradingLedger(path)
    first = _row(1)
    ledger.append(first)

    writer_acquired = threading.Event()
    release_writer = threading.Event()
    reader_done = threading.Event()
    reader_result = []
    reader_errors = []

    def hold_writer_lock():
        with exclusive_paper_ledger_write_lock(paper_ledger_write_lock_path(path)):
            writer_acquired.set()
            assert release_writer.wait(timeout=3.0)

    def read_ledger():
        try:
            reader_result.extend(ledger.records())
        except Exception as exc:  # pragma: no cover - asserted below when present.
            reader_errors.append(exc)
        finally:
            reader_done.set()

    writer = threading.Thread(target=hold_writer_lock, daemon=True)
    writer.start()
    assert writer_acquired.wait(timeout=3.0)

    reader = threading.Thread(target=read_ledger, daemon=True)
    reader.start()

    # A verified read must not inspect the ledger/checkpoint pair while the exclusive
    # writer commit section is held.
    assert not reader_done.wait(timeout=0.10)

    release_writer.set()
    writer.join(timeout=3.0)
    reader.join(timeout=3.0)

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert reader_errors == []
    assert reader_result == [first]
