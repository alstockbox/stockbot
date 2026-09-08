import numpy as np
import pandas as pd

from stockbot.paper.arena import PaperArenaCriteria, evaluate_paper_track
from stockbot.paper.ledger import PaperObservation, PaperTradingLedger


def _observations(strategy_id: str, sessions: int, *, weak: bool = False):
    dates = pd.date_range("2026-01-02", periods=sessions, freq="B", tz="UTC")
    rows = []
    for i, dt in enumerate(dates):
        if weak:
            net = -0.0005 + 0.001 * np.sin(i / 5.0)
            benchmark = 0.0002
            fill = 0.75
        else:
            net = 0.0012 + 0.0008 * np.sin(i / 7.0)
            benchmark = 0.0001
            fill = 0.99
        rows.append(
            PaperObservation(
                strategy_id=strategy_id,
                timestamp=dt.isoformat(),
                net_return=float(net),
                benchmark_return=float(benchmark),
                turnover=0.25,
                cost_rate=0.0001,
                fill_rate=fill,
                signal_count=4,
                regime="neutral_chop",
            )
        )
    return rows


def test_strong_forward_paper_track_can_become_live_evidence_eligible():
    criteria = PaperArenaCriteria(
        min_sessions=60,
        min_live_span_days=45,
        min_sharpe=0.0,
        min_excess_cagr=0.0,
        max_drawdown=0.30,
        min_fill_rate=0.90,
        max_average_turnover=1.0,
        min_bootstrap_confidence=0.60,
        reject_drift=False,
    )
    report = evaluate_paper_track(_observations("strategy-a", 90), criteria=criteria)

    assert report.sessions == 90
    assert report.live_span_days >= 45
    assert report.average_fill_rate > 0.95
    assert report.bootstrap_report is not None
    assert report.excess_cagr > 0.0
    assert report.live_eligible
    assert report.reasons == ()


def test_short_or_low_fill_paper_track_is_blocked():
    report = evaluate_paper_track(_observations("strategy-b", 20, weak=True))
    assert not report.live_eligible
    assert "insufficient_paper_sessions" in report.reasons
    assert "paper_fill_quality" in report.reasons


def test_paper_ledger_is_append_only_and_rejects_duplicate_session(tmp_path):
    ledger = PaperTradingLedger(tmp_path / "paper.jsonl")
    row = _observations("strategy-a", 1)[0]
    ledger.append(row)
    assert ledger.records(strategy_id="strategy-a") == [row]
    assert ledger.strategy_ids() == ("strategy-a",)

    try:
        ledger.append(row)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("expected duplicate observation to fail")
