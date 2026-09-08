from __future__ import annotations

import argparse

from stockbot.paper.arena import PaperArenaCriteria, evaluate_paper_track
from stockbot.paper.ledger import PaperTradingLedger, make_paper_observation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append and evaluate StockBot shadow/paper evidence. No broker execution is available."
    )
    parser.add_argument("--ledger", default="paper_memory/observations.jsonl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Append one realized shadow/paper session")
    record.add_argument("--strategy-id", required=True)
    record.add_argument("--net-return", type=float, required=True)
    record.add_argument("--benchmark-return", type=float, default=0.0)
    record.add_argument("--turnover", type=float, default=0.0)
    record.add_argument("--cost-rate", type=float, default=0.0)
    record.add_argument("--fill-rate", type=float, default=1.0)
    record.add_argument("--signal-count", type=int, default=0)
    record.add_argument("--regime", default=None)
    record.add_argument("--timestamp", default=None)
    record.add_argument("--notes", default=None)

    status = subparsers.add_parser("status", help="Evaluate accumulated forward paper evidence")
    status.add_argument("--strategy-id", required=True)
    status.add_argument("--min-sessions", type=int, default=60)
    status.add_argument("--min-span-days", type=int, default=45)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    ledger = PaperTradingLedger(args.ledger)
    if args.command == "record":
        observation = make_paper_observation(
            strategy_id=args.strategy_id,
            net_return=args.net_return,
            benchmark_return=args.benchmark_return,
            turnover=args.turnover,
            cost_rate=args.cost_rate,
            fill_rate=args.fill_rate,
            signal_count=args.signal_count,
            regime=args.regime,
            timestamp=args.timestamp,
            notes=args.notes,
        )
        ledger.append(observation)
        print(f"paper_recorded={observation.strategy_id}:{observation.timestamp}")
        print("broker_execution=disabled")
        return 0

    if args.command == "status":
        records = ledger.records(strategy_id=args.strategy_id)
        if not records:
            raise ValueError("no paper observations found for strategy")
        criteria = PaperArenaCriteria(
            min_sessions=args.min_sessions,
            min_live_span_days=args.min_span_days,
        )
        report = evaluate_paper_track(records, criteria=criteria)
        print(f"strategy_id={report.strategy_id}")
        print(f"sessions={report.sessions}")
        print(f"live_span_days={report.live_span_days}")
        print(f"paper_sharpe={report.metrics.get('sharpe', float('nan')):.6f}")
        print(f"paper_cagr={report.metrics.get('cagr', float('nan')):.6f}")
        print(f"paper_excess_cagr={report.excess_cagr:.6f}")
        print(f"paper_max_drawdown={report.metrics.get('max_drawdown', float('nan')):.6f}")
        print(f"average_fill_rate={report.average_fill_rate:.6f}")
        print(f"evidence_score={report.evidence_score:.6f}")
        print(f"live_evidence_eligible={'yes' if report.live_eligible else 'no'}")
        print("reasons=" + (",".join(report.reasons) if report.reasons else "none"))
        print("broker_execution=disabled")
        return 0

    raise ValueError(f"unsupported command: {args.command}")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 2
