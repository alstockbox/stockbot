from datetime import datetime, timezone
import json

import pytest

from stockbot.swing.paper_ledger import PaperTrade, SEKPaperLedger

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)


def trade(decision_id: str, side: str, *, qty: int = 2, price: float = 100.0, fee: float = 1.0):
    return PaperTrade(decision_id, NOW, "AAA", side, qty, price, fee)


def test_paper_buy_sell_accounting_and_restart(tmp_path):
    path = tmp_path / "stockbot-sek-paper.jsonl"
    book = SEKPaperLedger(path, initial_cash_sek=1_000.0)
    one = book.record_paper_fill(trade("one", "BUY"))
    assert one.cash_sek == pytest.approx(799.0)
    assert one.positions == {"AAA": 2}
    assert one.equity_sek is None
    two = book.snapshot({"AAA": 103.0})
    assert two.equity_sek == pytest.approx(1_005.0)
    second = SEKPaperLedger(path, initial_cash_sek=1_000.0)
    assert second.snapshot({"AAA": 100.0}).cash_sek == pytest.approx(799.0)
    closed = second.record_paper_fill(trade("two", "SELL", price=105.0))
    assert closed.positions == {}
    assert closed.cash_sek == pytest.approx(1008.0)
    assert closed.equity_sek == pytest.approx(1008.0)
    assert closed.last_sequence == 2


def test_fail_closed_on_overspend_duplicate_and_naked_short(tmp_path):
    book = SEKPaperLedger(tmp_path / "paper.jsonl")
    with pytest.raises(ValueError, match="INSUFFICIENT"):
        book.record_paper_fill(trade("a", "BUY", qty=20, price=100.0))
    with pytest.raises(ValueError, match="NAKED"):
        book.record_paper_fill(trade("a", "SELL"))
    book.record_paper_fill(trade("a", "BUY"))
    with pytest.raises(ValueError, match="DUPLICATE"):
        book.record_paper_fill(trade("a", "BUY"))
    with pytest.raises(ValueError, match="NAKED"):
        book.record_paper_fill(trade("b", "SELL", qty=3))
    assert book.snapshot().last_sequence == 1


def test_tampering_and_inconsistent_capital_fail_closed(tmp_path):
    path = tmp_path / "paper.jsonl"
    book = SEKPaperLedger(path)
    book.record_paper_fill(trade("first", "BUY"))
    with pytest.raises(ValueError, match="sequence"):
        SEKPaperLedger(path, initial_cash_sek=2000).snapshot()
    entry = json.loads(path.read_text())
    entry["event"]["price_sek"] = 0.01
    path.write_text(json.dumps(entry) + "\n")
    with pytest.raises(ValueError, match="hash"):
        book.snapshot()


def test_reject_invalid_timestamp_quantity_price_and_missing_marks(tmp_path):
    with pytest.raises(ValueError):
        PaperTrade("bad", datetime(2026, 10, 29), "AAA", "BUY", 1, 10, 0)
    with pytest.raises(ValueError):
        trade("bad", "BUY", qty=-1)
    with pytest.raises(ValueError):
        trade("bad", "BUY", price=float("nan"))
    book = SEKPaperLedger(tmp_path / "paper.jsonl")
    book.record_paper_fill(trade("a", "BUY"))
    with pytest.raises(ValueError, match="marks"):
        book.snapshot({"BBB": 100})
