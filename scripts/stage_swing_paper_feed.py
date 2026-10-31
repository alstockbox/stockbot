"""Stage or inspect SYNTHETIC XSTO EOD paper-feed fixtures, no broker writes.

This program expressly rejects real/third-party data. It does not publish,
schedule, trade, connect to Avanza or claim a real forward-paper session.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from stockbot.swing.paper_feed import PaperFeedJournal, PaperSession


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic XSTO paper-feed REHEARSAL ONLY")
    parser.add_argument("--journal", type=Path, required=True,
                        help="Local JSONL synthetic paper observation journal")
    parser.add_argument("--session-json", type=Path,
                        help="Optional single synthetic completed session JSON")
    args = parser.parse_args()
    journal = PaperFeedJournal(args.journal)
    if args.session_json is not None:
        if not args.session_json.is_file():
            parser.error("synthetic session JSON not found")
        raw = json.loads(args.session_json.read_text(encoding="utf-8"))
        event = PaperSession.from_dict(raw)
        journal.stage(event, clock_utc=datetime.now(timezone.utc))
    summary = asdict(journal.snapshot())
    summary["readiness"] = asdict(journal.readiness())
    summary["message"] = (
        "SYNTHETIC ONLY. No actual data-license verification, "
        "real forward paper, market orders, funding or broker connectivity."
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
