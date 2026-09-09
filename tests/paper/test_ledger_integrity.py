from dataclasses import asdict
import json

import pytest

from stockbot.paper.ledger import PaperObservation, PaperTradingLedger


def _row(day: int, net_return: float = 0.001) -> PaperObservation:
    return PaperObservation(
        strategy_id="strategy-integrity",
        timestamp=f"2026-09-{day:02d}T16:00:00+00:00",
        net_return=net_return,
        turnover=0.2,
        cost_rate=0.0001,
        fill_rate=0.99,
        signal_count=3,
    )


def _payloads(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_payloads(path, payloads):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in payloads), encoding="utf-8")


def test_paper_ledger_new_rows_are_hash_chained_and_verified(tmp_path):
    path = tmp_path / "paper.jsonl"
    ledger = PaperTradingLedger(path)
    rows = [_row(1), _row(2), _row(3)]
    for row in rows:
        ledger.append(row)

    payloads = _payloads(path)
    assert all(payload["__ledger_schema_version"] == 1 for payload in payloads)
    assert payloads[1]["__ledger_previous_hash"] == payloads[0]["__ledger_record_hash"]
    assert payloads[2]["__ledger_previous_hash"] == payloads[1]["__ledger_record_hash"]
    assert ledger.records() == rows


def test_paper_ledger_detects_historical_row_tampering(tmp_path):
    path = tmp_path / "paper.jsonl"
    ledger = PaperTradingLedger(path)
    for day in (1, 2, 3):
        ledger.append(_row(day))

    payloads = _payloads(path)
    payloads[0]["net_return"] = 0.75
    _write_payloads(path, payloads)

    with pytest.raises(ValueError, match="integrity"):
        ledger.records()


def test_paper_ledger_detects_deleted_or_reordered_rows(tmp_path):
    path = tmp_path / "paper.jsonl"
    ledger = PaperTradingLedger(path)
    for day in (1, 2, 3):
        ledger.append(_row(day))

    payloads = _payloads(path)
    _write_payloads(path, [payloads[0], payloads[2]])
    with pytest.raises(ValueError, match="integrity"):
        ledger.records()

    _write_payloads(path, [payloads[1], payloads[0], payloads[2]])
    with pytest.raises(ValueError, match="integrity"):
        ledger.records()


def test_first_chained_append_anchors_legacy_prefix(tmp_path):
    path = tmp_path / "paper.jsonl"
    legacy = _row(1)
    path.write_text(json.dumps(asdict(legacy), sort_keys=True) + "\n", encoding="utf-8")
    ledger = PaperTradingLedger(path)

    current = _row(2)
    ledger.append(current)
    assert ledger.records() == [legacy, current]

    payloads = _payloads(path)
    assert "__ledger_record_hash" not in payloads[0]
    assert payloads[1]["__ledger_previous_hash"]

    payloads[0]["net_return"] = 0.5
    _write_payloads(path, payloads)
    with pytest.raises(ValueError, match="integrity"):
        ledger.records()
