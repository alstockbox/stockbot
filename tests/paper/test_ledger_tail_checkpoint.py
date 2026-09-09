from dataclasses import asdict
import json
from pathlib import Path

import pytest

import stockbot.paper.ledger as ledger_module
from stockbot.paper.ledger import PaperObservation, PaperTradingLedger


def _row(day: int, *, strategy_id: str = "strategy-tail") -> PaperObservation:
    return PaperObservation(
        strategy_id=strategy_id,
        timestamp=f"2026-09-{day:02d}T16:00:00+00:00",
        net_return=0.001,
        turnover=0.2,
        cost_rate=0.0001,
        fill_rate=0.99,
        signal_count=3,
    )


def _payloads(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _head_path(path: Path) -> Path:
    return Path(str(path) + ".head.json")


def test_new_paper_ledger_rows_require_committed_head_checkpoint(tmp_path):
    path = tmp_path / "paper.jsonl"
    ledger = PaperTradingLedger(path)
    ledger.append(_row(1))
    ledger.append(_row(2))

    payloads = _payloads(path)
    assert [payload["__ledger_schema_version"] for payload in payloads] == [2, 2]

    checkpoint = json.loads(_head_path(path).read_text(encoding="utf-8"))
    assert checkpoint["schema_version"] == 1
    assert checkpoint["row_count"] == 2
    assert checkpoint["tail_hash"] == payloads[-1]["__ledger_record_hash"]
    assert checkpoint["ledger_size_bytes"] == path.stat().st_size
    assert ledger.records() == [_row(1), _row(2)]


def test_paper_ledger_detects_tail_truncation_against_checkpoint(tmp_path):
    path = tmp_path / "paper.jsonl"
    ledger = PaperTradingLedger(path)
    for day in (1, 2, 3):
        ledger.append(_row(day))

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint"):
        ledger.records()


def test_paper_ledger_v2_fails_closed_when_head_checkpoint_is_missing(tmp_path):
    path = tmp_path / "paper.jsonl"
    ledger = PaperTradingLedger(path)
    ledger.append(_row(1))
    _head_path(path).unlink()

    with pytest.raises(ValueError, match="checkpoint"):
        ledger.records()


def test_v1_chain_without_checkpoint_remains_readable_and_upgrades_on_append(tmp_path):
    path = tmp_path / "paper.jsonl"
    legacy = _row(1)
    observation_payload = asdict(legacy)
    previous_hash = ledger_module._LEDGER_GENESIS_HASH
    record_hash = ledger_module._next_ledger_hash(previous_hash, observation_payload)
    payload = dict(observation_payload)
    payload["__ledger_schema_version"] = 1
    payload["__ledger_previous_hash"] = previous_hash
    payload["__ledger_record_hash"] = record_hash
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    ledger = PaperTradingLedger(path)
    assert ledger.records() == [legacy]
    assert not _head_path(path).exists()

    current = _row(2)
    ledger.append(current)
    payloads = _payloads(path)
    assert payloads[0]["__ledger_schema_version"] == 1
    assert payloads[1]["__ledger_schema_version"] == 2
    assert _head_path(path).is_file()
    assert ledger.records() == [legacy, current]
