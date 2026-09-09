from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from stockbot.data.research_inputs import load_point_in_time_feature_store
from stockbot.data.snapshots import SnapshotStore
from stockbot.paper.arena import PaperArenaCriteria, evaluate_paper_track
from stockbot.paper.ledger import PaperTradingLedger, make_paper_observation
from stockbot.paper.runner import load_frozen_shadow_artifact, run_shadow_step


def _parse_names(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("auxiliary features must contain unique non-empty names")
    return values


def _add_shadow_step_arguments(parser: argparse.ArgumentParser, *, explicit_snapshot: bool) -> None:
    parser.add_argument("--artifact", required=True, help="Frozen shadow artifact directory")
    parser.add_argument("--snapshot-root", required=True, help="Root directory containing immutable StockBot snapshots")
    if explicit_snapshot:
        parser.add_argument("--snapshot-id", required=True, help="Immutable StockBot snapshot ID to process")
    parser.add_argument("--state", default="paper_memory/shadow-state.json", help="Atomic frozen-runner state JSON")
    parser.add_argument("--data-age-seconds", type=float, default=0.0)
    parser.add_argument("--kill-switch", action="store_true", help="Force the hard risk engine to zero all next targets")
    parser.add_argument("--auxiliary-input", default=None, help="Fingerprint-verified point-in-time auxiliary feature store")
    parser.add_argument("--auxiliary-features", default=None, help="Optional comma-separated auxiliary feature subset")
    parser.add_argument("--auxiliary-max-age-days", type=int, default=None)
    parser.add_argument("--auxiliary-min-coverage", type=float, default=0.80)


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
        help="Advance a hash-verified frozen strategy through one explicit immutable market snapshot",
    )
    _add_shadow_step_arguments(step, explicit_snapshot=True)

    auto_step = subparsers.add_parser(
        "auto-step",
        help="Advance a frozen strategy through the newest verified compatible immutable snapshot",
    )
    _add_shadow_step_arguments(auto_step, explicit_snapshot=False)

    status = subparsers.add_parser("status", help="Evaluate accumulated forward paper evidence")
    status.add_argument("--strategy-id", required=True)
    status.add_argument("--min-sessions", type=int, default=60)
    status.add_argument("--min-span-days", type=int, default=45)
    return parser


def _validate_shadow_args(args: argparse.Namespace):
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
    return auxiliary_names, auxiliary_store


def _parse_snapshot_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("snapshot observation timestamp must be timezone-aware")
    return parsed


def _select_latest_compatible_snapshot(snapshot_root: str | Path, artifact_symbols: tuple[str, ...]):
    root = Path(snapshot_root)
    if not root.is_dir():
        raise ValueError("no compatible verified snapshots found")
    snapshot_ids = tuple(sorted(child.name for child in root.iterdir() if child.is_dir()))
    if not snapshot_ids:
        raise ValueError("no compatible verified snapshots found")

    expected_symbols = tuple(sorted(str(symbol).upper() for symbol in artifact_symbols))
    store = SnapshotStore(root)
    compatible = []
    for snapshot_id in snapshot_ids:
        # Deliberately verify every snapshot directory before using it. A corrupt
        # immutable snapshot is an integrity event, not a reason to silently roll
        # backward to older market data.
        snapshot = store.load(snapshot_id)
        if tuple(snapshot.manifest.symbols) != expected_symbols:
            continue
        if not snapshot.manifest.last_observation:
            raise ValueError("compatible snapshot has no last-observation evidence")
        latest_market_time = max(
            _parse_snapshot_time(value)
            for value in snapshot.manifest.last_observation.values()
        )
        compatible.append(
            (
                latest_market_time,
                snapshot.manifest.created_at,
                snapshot.snapshot_id,
                snapshot,
            )
        )

    if not compatible:
        raise ValueError("no compatible verified snapshots found")
    return max(compatible, key=lambda item: (item[0], item[1], item[2]))[3]


def _run_shadow_snapshot(args: argparse.Namespace, snapshot) -> int:
    auxiliary_names, auxiliary_store = _validate_shadow_args(args)
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
        print("paper_observation=pending_or_idempotent")
    else:
        print(f"paper_observation={result.observation.timestamp}")
        print(f"paper_net_return={result.observation.net_return:.6f}")
        print(f"paper_fill_rate={result.observation.fill_rate:.6f}")
        print(f"paper_cost_rate={result.observation.cost_rate:.6f}")
    print("broker_execution=disabled")
    return 0


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
        snapshot = SnapshotStore(args.snapshot_root).load(args.snapshot_id)
        return _run_shadow_snapshot(args, snapshot)

    if args.command == "auto-step":
        artifact = load_frozen_shadow_artifact(args.artifact)
        snapshot = _select_latest_compatible_snapshot(args.snapshot_root, artifact.symbols)
        print(f"selected_snapshot_id={snapshot.snapshot_id}")
        print(f"selected_snapshot_fingerprint={snapshot.manifest.dataset_fingerprint}")
        return _run_shadow_snapshot(args, snapshot)

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
