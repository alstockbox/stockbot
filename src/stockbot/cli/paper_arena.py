from __future__ import annotations

import argparse

from stockbot.data.research_inputs import load_point_in_time_feature_store
from stockbot.data.snapshots import SnapshotStore
from stockbot.paper.arena import PaperArenaCriteria, evaluate_paper_track
from stockbot.paper.ledger import PaperTradingLedger, make_paper_observation
from stockbot.paper.runner import run_shadow_step


def _parse_names(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("auxiliary features must contain unique non-empty names")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append, advance and evaluate StockBot shadow/paper evidence. No broker execution is available."
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

    step = subparsers.add_parser(
        "step",
        help="Advance a hash-verified frozen strategy through one immutable market snapshot",
    )
    step.add_argument("--artifact", required=True, help="Frozen shadow artifact directory")
    step.add_argument("--snapshot-root", required=True, help="Root directory containing immutable StockBot snapshots")
    step.add_argument("--snapshot-id", required=True, help="Immutable StockBot snapshot ID to process")
    step.add_argument("--state", default="paper_memory/shadow-state.json", help="Atomic frozen-runner state JSON")
    step.add_argument("--data-age-seconds", type=float, default=0.0)
    step.add_argument("--kill-switch", action="store_true", help="Force the hard risk engine to zero all next targets")
    step.add_argument("--auxiliary-input", default=None, help="Fingerprint-verified point-in-time auxiliary feature store")
    step.add_argument("--auxiliary-features", default=None, help="Optional comma-separated auxiliary feature subset")
    step.add_argument("--auxiliary-max-age-days", type=int, default=None)
    step.add_argument("--auxiliary-min-coverage", type=float, default=0.80)

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

    if args.command == "step":
        if args.data_age_seconds < 0.0:
            raise ValueError("data age seconds cannot be negative")
        if args.auxiliary_max_age_days is not None and args.auxiliary_max_age_days < 0:
            raise ValueError("auxiliary max age days cannot be negative")
        if not 0.0 < args.auxiliary_min_coverage <= 1.0:
            raise ValueError("auxiliary minimum coverage must be in (0,1]")
        auxiliary_names = _parse_names(args.auxiliary_features)
        if auxiliary_names is not None and not args.auxiliary_input:
            raise ValueError("auxiliary features require --auxiliary-input")
        auxiliary_store = (
            None
            if not args.auxiliary_input
            else load_point_in_time_feature_store(args.auxiliary_input)
        )
        snapshot = SnapshotStore(args.snapshot_root).load(args.snapshot_id)
        result = run_shadow_step(
            snapshot.bars,
            artifact_dir=args.artifact,
            state_path=args.state,
            ledger_path=args.ledger,
            snapshot_fingerprint=snapshot.manifest.dataset_fingerprint,
            data_age_seconds=args.data_age_seconds,
            kill_switch=args.kill_switch,
            auxiliary_store=auxiliary_store,
            auxiliary_feature_names=auxiliary_names,
            auxiliary_max_age_days=args.auxiliary_max_age_days,
            auxiliary_min_coverage=args.auxiliary_min_coverage,
        )
        print(f"shadow_strategy_id={result.strategy_id}")
        print(f"shadow_artifact_id={result.artifact_id}")
        print(f"processed_timestamp={result.processed_timestamp}")
        print(f"pending_signal_timestamp={result.pending_signal_timestamp}")
        print(f"target_count={sum(abs(float(value)) > 1e-12 for value in result.target_weights.values())}")
        if result.observation is None:
            print("paper_observation=pending_first_forward_session")
        else:
            print(f"paper_observation={result.observation.timestamp}")
            print(f"paper_net_return={result.observation.net_return:.6f}")
            print(f"paper_fill_rate={result.observation.fill_rate:.6f}")
            print(f"paper_cost_rate={result.observation.cost_rate:.6f}")
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
