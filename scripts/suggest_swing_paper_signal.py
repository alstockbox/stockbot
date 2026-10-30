"""Explicit, SYNTHETIC ONLY paper swing signal preview/staging.

Preview is default and does not generate a decision. --stage-synthetic
persists one broker-free synthetic intent only after as-of verification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from stockbot.swing.paper_rehearsal import SyntheticPaperWorkflow
from stockbot.swing.paper_signal import SyntheticSwingSignal


def main() -> int:
    parser=argparse.ArgumentParser(
        description="Synthetic XSTO research candidate; no real data, paper, or live orders."
    )
    parser.add_argument("--feed",type=Path,required=True)
    parser.add_argument("--intents",type=Path,required=True)
    parser.add_argument("--ledger",type=Path,required=True)
    parser.add_argument("--initial-sek",type=float,default=1000.0)
    parser.add_argument(
        "--stage-synthetic",action="store_true",
        help="Explicitly persist a SYNTHETIC intent; cannot stage old decisions."
    )
    args=parser.parse_args()
    wf=SyntheticPaperWorkflow(
        feed_path=args.feed,intents_path=args.intents,
        fills_path=args.ledger,initial_sek=args.initial_sek
    )
    signal=SyntheticSwingSignal(wf)
    clock=datetime.now(timezone.utc)
    recommendation=(
        signal.stage(clock_utc=clock) if args.stage_synthetic
        else signal.preview(clock_utc=clock)
    )
    report=recommendation.to_dict()
    report["note"]=(
        "SYNTHETIC ONLY: No licensed market data, intraday stop routing, "
        "genuine forward paper, or broker connection."
    )
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
