from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from stockbot.brokers.snapshot_feed import SnapshotFileProvider
from stockbot.opportunity.ranker import Opportunity, OpportunityPolicy
from stockbot.runtime.shadow_runtime import ShadowTradePlan, ShadowTradingRuntime


def _write_demo_snapshot(path: Path, symbol: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "account": {
            "balance": 466.0,
            "equity": 466.0,
            "free_margin": 466.0,
            "currency": "USD",
            "timestamp": now,
        },
        "symbols": {
            symbol: {
                "quote": {
                    "bid": 1.1000,
                    "ask": 1.1002,
                    "timestamp": now,
                },
                "spec": {
                    "tick_size": 0.00001,
                    "tick_value": 1.0,
                    "volume_min": 0.01,
                    "volume_max": 100.0,
                    "volume_step": 0.01,
                    "trade_enabled": True,
                },
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one read-only StockBot shadow cycle. Never sends broker orders."
    )
    parser.add_argument("--snapshot-file", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--expected-return", type=float, default=0.01)
    parser.add_argument("--risk-fraction", type=float, default=0.003)
    parser.add_argument("--notional-fraction", type=float, default=0.25)
    parser.add_argument("--journal", type=Path, default=Path("var/shadow_journal.jsonl"))
    return parser


def _run(snapshot_file: Path, args: argparse.Namespace) -> int:
    provider = SnapshotFileProvider(snapshot_file)
    runtime = ShadowTradingRuntime(
        provider,
        args.journal,
        opportunity_policy=OpportunityPolicy(minimum_net_edge=0.0005),
    )
    now = datetime.now(timezone.utc)
    decision_id = f"shadow-{args.symbol}-{int(now.timestamp() * 1000)}"
    opportunity = Opportunity(
        symbol=args.symbol,
        source="shadow_cli_probe",
        expected_return=args.expected_return,
        expected_cost=0.0,
        downside_risk=0.001,
        correlation_penalty=0.001,
        uncertainty=0.001,
        confidence=0.80,
    )
    result = runtime.run_cycle(
        [
            ShadowTradePlan(
                decision_id,
                opportunity,
                risk_fraction_of_equity=args.risk_fraction,
                notional_fraction_of_equity=args.notional_fraction,
            )
        ],
        timestamp=now,
        regime="shadow_probe",
    )
    print(
        json.dumps(
            {
                "source": str(snapshot_file),
                "accepted": result.accepted,
                "rejected": result.rejected,
                "open_shadow_positions": len(runtime.executor.positions),
                "live_orders_possible": False,
                "decisions": [
                    {
                        "decision_id": decision.decision_id,
                        "approved": decision.approved,
                        "reasons": decision.reasons,
                    }
                    for decision in result.decisions
                ],
            },
            sort_keys=True,
        )
    )
    print("SHADOW_CYCLE_OK")
    return 0


def main() -> int:
    args = _build_parser().parse_args()
    if args.demo:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "mt5_snapshot.json"
            _write_demo_snapshot(snapshot, args.symbol)
            return _run(snapshot, args)
    if args.snapshot_file is None:
        print(
            "SHADOW_FEED_NOT_CONFIGURED: use --demo or --snapshot-file PATH. "
            "No broker write capability exists in this command."
        )
        return 2
    return _run(args.snapshot_file, args)


if __name__ == "__main__":
    raise SystemExit(main())
