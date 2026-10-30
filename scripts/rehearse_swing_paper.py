"""Offline synthetic EOD -> next-open rehearsal CLI (no actual paper/live)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from stockbot.swing.paper_rehearsal import SyntheticPaperWorkflow


def main() -> int:
    parser=argparse.ArgumentParser(
        description="SYNTHETIC XSTO OPEN RECONSTRUCTION ONLY; no brokerage, orders or real data."
    )
    parser.add_argument("--feed",type=Path,required=True)
    parser.add_argument("--intents",type=Path,required=True)
    parser.add_argument("--ledger",type=Path,required=True)
    parser.add_argument("--initial-sek",type=float,default=1000.0)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan",action="store_true",
                      help="Record one synthetic decision for next XSTO session")
    mode.add_argument("--reconcile",action="store_true",
                      help="Reconstruct synthetic fills from already observed next-day OHLCV")
    parser.add_argument("--symbol",help="Synthetic instrument symbol, when planning")
    parser.add_argument("--side",choices=("BUY","SELL"),help="Synthetic action, when planning")
    parser.add_argument("--quantity",type=int,help="Whole share quantity, when planning")
    args=parser.parse_args()
    if args.plan and (not args.symbol or not args.side or args.quantity is None):
        parser.error("--plan requires --symbol, --side and --quantity")
    if args.reconcile and (args.symbol or args.side or args.quantity is not None):
        parser.error("--reconcile must not have a new trading decision")
    workflow=SyntheticPaperWorkflow(
        feed_path=args.feed, intents_path=args.intents,
        fills_path=args.ledger, initial_sek=args.initial_sek
    )
    clock=datetime.now(timezone.utc)
    if args.plan:
        decision_id=workflow.plan(
            symbol=args.symbol, side=args.side, quantity=args.quantity,
            clock_utc=clock
        )
        report={
            "mode":"SYNTHETIC_INTENT_ONLY",
            "decision_id":decision_id,
            "real_forward_paper_started":False,
            "broker_orders_sent":0,
        }
    else:
        records=workflow.reconcile(clock_utc=clock)
        report={
            "mode":"SYNTHETIC_RETROSPECTIVE_NEXT_XSTO_OPEN",
            "reconciled":[asdict(r) for r in records],
            "real_forward_paper_started":False,
            "broker_orders_sent":0,
        }
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
