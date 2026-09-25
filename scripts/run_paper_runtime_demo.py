from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from stockbot.execution.safeguard import SafeguardState
from stockbot.opportunity.ranker import Opportunity, OpportunityPolicy
from stockbot.runtime.paper_runtime import PaperTradePlan, PaperTradingRuntime


def main() -> int:
    plans = [
        PaperTradePlan(
            "demo-a",
            Opportunity("EURUSD", "trend", 0.010, 0.001, 0.002, 0.001, 0.001, 0.85),
            risk_fraction_of_equity=0.003,
            notional_fraction_of_equity=0.25,
        ),
        PaperTradePlan(
            "demo-b",
            Opportunity("GBPUSD", "breakout", 0.008, 0.001, 0.002, 0.001, 0.001, 0.80),
            risk_fraction_of_equity=0.003,
            notional_fraction_of_equity=0.25,
        ),
        PaperTradePlan(
            "demo-c",
            Opportunity("XAUUSD", "weak", 0.002, 0.001, 0.003, 0.001, 0.001, 0.60),
            risk_fraction_of_equity=0.003,
            notional_fraction_of_equity=0.20,
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        journal = Path(tmp) / "paper-journal.jsonl"
        runtime = PaperTradingRuntime(
            journal,
            opportunity_policy=OpportunityPolicy(minimum_net_edge=0.0005),
        )
        result = runtime.run_cycle(
            plans,
            SafeguardState(
                equity_usd=466.0,
                equity_peak_usd=466.0,
                data_age_seconds=1.0,
            ),
            timestamp=datetime.now(timezone.utc),
            regime="neutral_chop",
        )
        payload = {
            "ranked": len(result.ranked),
            "accepted": result.accepted,
            "rejected": result.rejected,
            "journal_entries": len(runtime.journal.entries()),
        }
        print(json.dumps(payload, sort_keys=True))
        if result.accepted < 1 or payload["journal_entries"] != len(result.ranked):
            return 1
    print("PAPER_RUNTIME_DEMO_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
