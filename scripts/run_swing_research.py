"""CLI for StockBot's SEK swing *research* simulator, never places orders.

Usage: python scripts/run_swing_research.py --csv path/to/ohlcv.csv
Input: date,symbol,open,high,low,close,volume; at least 205 aligned days.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stockbot.swing.engine import SwingConfig, run_swing_backtest


def main() -> int:
    parser = argparse.ArgumentParser(description="Swedish equity swing research, NO BROKER ORDERS")
    parser.add_argument("--csv", type=Path, required=True, help="License-compliant daily OHLCV CSV")
    parser.add_argument("--capital-sek", type=float, default=1_000.0)
    parser.add_argument("--output", type=Path, help="Optional JSON report, overwritten if present")
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error("CSV file does not exist")
    frame = pd.read_csv(args.csv)
    result = run_swing_backtest(frame, SwingConfig(initial_sek=args.capital_sek))
    report = {
        "mode": "BACKTEST_ONLY",
        "orders_sent": 0,
        "research_grade": "UNVERIFIED",
        "warning": "Price data licensing, split/dividends, price availability, survivorship and real fills not validated.",
        "capital_initial_sek": args.capital_sek,
        "capital_final_sek": round(result.final_equity_sek, 2),
        "return_pct": round((result.final_equity_sek / args.capital_sek - 1) * 100, 3),
        "max_drawdown_pct": round(result.max_drawdown * 100, 3),
        "commission_sek": round(result.total_fees_sek, 2),
        "execution_impact_sek": round(result.total_execution_impact_sek, 2),
        "hypothetical_fills": len(result.fills),
        "first_session": str(result.equity.index[0].date()),
        "last_session": str(result.equity.index[-1].date()),
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
