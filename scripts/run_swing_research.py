"""Offline SEK swing research CLI. No broker login, deposits or orders.

Usage:
 python scripts/run_swing_research.py --csv bars.csv --manifest provenance.json
Data: date,symbol,open,high,low,close,volume; >=205 aligned XSTO sessions.
Provenance manifest does NOT independently validate data licensing or returns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stockbot.swing.data_quality import XSTOProvenance, audit_xsto_data
from stockbot.swing.benchmark import IndexProvenance, compare_oos_to_index
from stockbot.swing.universe import parse_universe_records, audit_instrument_history
from stockbot.swing.corporate_actions_io import parse_action_records
from stockbot.swing.engine import SwingConfig, run_swing_backtest
from stockbot.swing.walkforward import WalkForwardConfig, run_walkforward


def main() -> int:
    parser = argparse.ArgumentParser(description="XSTO swing research ONLY: no broker connectivity")
    parser.add_argument("--csv", type=Path, required=True, help="License-compliant OHLCV CSV")
    parser.add_argument("--manifest", type=Path, help="JSON data provenance; enables XSTO calendar audit")
    parser.add_argument("--capital-sek", type=float, default=1_000.0)
    parser.add_argument("--min-prior-turnover-sek", type=float, default=100_000.0,
                        help="Minimum 20-session median SEK daily turnover, previous close only")
    parser.add_argument("--max-daily-volume-fraction", type=float, default=0.02,
                        help="Fraction of reported daily share volume available for hypothetical fills")
    parser.add_argument("--actions-json", type=Path, help="Optional audited corporate action events JSON; RAW prices only")
    parser.add_argument("--universe-json", type=Path, help="Optional exact historical instrument-lifecycle manifest")
    parser.add_argument("--benchmark-csv", type=Path, help="Authorized Swedish GI index date,close CSV; requires --walkforward")
    parser.add_argument("--benchmark-manifest", type=Path, help="Index rights and collection provenance JSON")
    parser.add_argument("--walkforward", action="store_true", help="Train-only tuning and held-out rolling evaluation; requires --manifest and >=882 sessions")
    parser.add_argument("--output", type=Path, help="Optional JSON report, overwritten if present")
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error("CSV file does not exist")
    frame = pd.read_csv(args.csv)
    if bool(args.benchmark_csv) != bool(args.benchmark_manifest):
        parser.error("--benchmark-csv and --benchmark-manifest must be provided together")
    if args.benchmark_csv is not None and not args.walkforward:
        parser.error("benchmark comparison requires --walkforward")
    if args.universe_json is not None and args.manifest is None:
        parser.error("--universe-json requires --manifest")
    receipt = None
    universe_receipt = None
    if args.manifest is not None:
        if not args.manifest.is_file():
            parser.error("data manifest does not exist")
        raw = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            parser.error("data manifest must be a JSON object")
        provenance = XSTOProvenance.from_dict(raw)
        if args.universe_json is not None:
            if not args.universe_json.is_file():
                parser.error("universe JSON file does not exist")
            universe_receipt = audit_instrument_history(
                frame,
                parse_universe_records(json.loads(
                    args.universe_json.read_text(encoding="utf-8")
                )),
                collected_at_utc=provenance.collected_at_utc,
            )
            if not universe_receipt.compatible_with_current_fixed_panel_engine:
                parser.error(
                    "historical IPO/delisting transitions detected: fixed-panel "
                    "backtest prohibited; run scripts/audit_swing_universe.py instead"
                )
        receipt = audit_xsto_data(frame, provenance)  # mandatory before backtest
    corporate_actions = ()
    if args.actions_json is not None:
        if receipt is None or provenance.price_basis != "RAW":
            parser.error("--actions-json requires an XSTO --manifest with price_basis=RAW")
        if not args.actions_json.is_file():
            parser.error("corporate actions JSON file does not exist")
        corporate_actions = parse_action_records(
            json.loads(args.actions_json.read_text(encoding="utf-8"))
        )
    if args.walkforward and receipt is None:
        parser.error("--walkforward requires --manifest for XSTO calendar audit")
    cfg = SwingConfig(initial_sek=args.capital_sek,
                      min_prior_median_turnover_sek=args.min_prior_turnover_sek,
                      max_daily_volume_fraction=args.max_daily_volume_fraction)
    if args.walkforward:
        wf = run_walkforward(frame, WalkForwardConfig(base=cfg),
                             corporate_actions=corporate_actions)
        report = wf.to_dict()
        report["data_quality"] = receipt.to_dict()  # type: ignore[union-attr]
        report["orders_sent"] = 0
        report["universe_audit"] = universe_receipt.to_dict() if universe_receipt else {
            "status": "NOT_AUDITED", "survivorship_independently_verified": False
        }
        if args.benchmark_csv is not None:
            for path in (args.benchmark_csv, args.benchmark_manifest):
                if not path.is_file():
                    parser.error(f"missing benchmark input: {path}")
            benchmark_manifest = IndexProvenance.from_dict(json.loads(
                args.benchmark_manifest.read_text(encoding="utf-8")
            ))
            report["benchmark_comparison"] = compare_oos_to_index(
                wf, pd.read_csv(args.benchmark_csv), benchmark_manifest
            ).to_dict()
        report["warning"] = "OOS backtest is not independent proof of tradability, profitability, data rights, actions or market impact."
        report["max_daily_volume_fraction"] = cfg.max_daily_volume_fraction
        report["min_prior_median_turnover_sek"] = cfg.min_prior_median_turnover_sek
    else:
        result = run_swing_backtest(frame, cfg,
                                    corporate_actions=corporate_actions)
        report = {
        "mode": "BACKTEST_ONLY",
        "orders_sent": 0,
        "research_grade": "UNVERIFIED",
        "universe_audit": universe_receipt.to_dict() if universe_receipt else {
            "status": "NOT_AUDITED", "survivorship_independently_verified": False
        },
        "data_quality": receipt.to_dict() if receipt else {
            "status": "NOT_AUDITED", "order_execution_permitted": False
        },
        "warning": "Data license, split/dividend events, delistings, historical universe, benchmark and executable fills are NOT independently validated.",
        "capital_initial_sek": args.capital_sek,
        "capital_final_sek": round(result.final_equity_sek, 2),
        "return_pct": round((result.final_equity_sek / args.capital_sek - 1) * 100, 3),
        "max_drawdown_pct": round(result.max_drawdown * 100, 3),
        "commission_sek": round(result.total_fees_sek, 2),
        "execution_impact_sek": round(result.total_execution_impact_sek, 2),
        "hypothetical_fills": len(result.fills),
        "liquidity_rejected_entries": result.liquidity_rejected_entries,
        "partial_exit_events": result.partial_exit_events,
        "unresolved_exit_shares": result.unresolved_exit_shares,
        "min_prior_median_turnover_sek": cfg.min_prior_median_turnover_sek,
        "max_daily_volume_fraction": cfg.max_daily_volume_fraction,
        "corporate_action_events": list(result.corporate_action_events),
        "dividends_paid_sek": round(result.dividend_cash_paid_sek, 2),
        "outstanding_dividend_receivable_sek": round(result.dividend_receivable_sek, 2),
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
