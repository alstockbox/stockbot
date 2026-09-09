from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pandas as pd

from stockbot.data.download import DownloadError, download_market_snapshot
from stockbot.data.providers.http import ProviderError
from stockbot.data.providers.tiingo import TiingoProvider
from stockbot.data.providers.yahoo_bootstrap import YahooBootstrapProvider
from stockbot.data.research_inputs import load_point_in_time_feature_store
from stockbot.data.schemas import DataGrade
from stockbot.data.snapshots import SnapshotStore
from stockbot.paper.arena import PaperArenaCriteria, evaluate_paper_track
from stockbot.paper.deployment_gate import (
    evaluate_bound_deployment_review,
    verify_research_evidence_bundle,
)
from stockbot.paper.ledger import PaperTradingLedger, make_paper_observation
from stockbot.paper.locking import exclusive_shadow_command_lock, shadow_command_lock_path
from stockbot.paper.provenance import verify_frozen_paper_provenance
from stockbot.paper.runner import load_frozen_shadow_artifact, run_shadow_step
from stockbot.research.quarantine import QuarantineConfig
from stockbot.research.quarantine_audit import (
    FrozenStrategySpec,
    load_verified_quarantine_audit_record,
    run_single_quarantine_audit,
)


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
    parser.add_argument(
        "--lock",
        default=None,
        help="Optional exclusive process-lock file; defaults to <state>.lock",
    )
    parser.add_argument(
        "--data-age-seconds",
        type=float,
        default=None,
        help="Optional explicit feed-age override; otherwise derived from verified snapshot retrieval time",
    )
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

    cycle = subparsers.add_parser(
        "cycle",
        help="Refresh immutable market data for the frozen universe and advance one shadow step",
    )
    _add_shadow_step_arguments(cycle, explicit_snapshot=False)
    cycle.add_argument("--provider", required=True, choices=("yahoo-bootstrap", "tiingo"))
    cycle.add_argument(
        "--start",
        default=None,
        help="Optional download start date; otherwise inherited from the latest verified compatible snapshot",
    )
    cycle.add_argument(
        "--end",
        default=None,
        help="Inclusive download end date; defaults to the current UTC calendar date",
    )
    cycle.add_argument(
        "--report",
        default=None,
        help="Optional atomic machine-readable JSON report for schedulers and monitoring",
    )

    status = subparsers.add_parser(
        "status",
        help="Evaluate forward paper evidence with frozen artifact/snapshot verification",
    )
    status.add_argument("--strategy-id", required=True)
    status.add_argument("--artifact", required=True, help="Frozen shadow artifact directory")
    status.add_argument(
        "--snapshot-root",
        required=True,
        help="Root directory containing immutable StockBot snapshots referenced by the paper ledger",
    )
    status.add_argument(
        "--report",
        default=None,
        help="Optional atomic machine-readable JSON status report for schedulers and monitoring",
    )
    status.add_argument("--min-sessions", type=int, default=60)
    status.add_argument("--min-span-days", type=int, default=45)

    quarantine_audit = subparsers.add_parser(
        "quarantine-audit",
        help=(
            "Run the one-time sealed-quarantine audit for a verified research-grade frozen champion"
        ),
    )
    quarantine_audit.add_argument("--artifact", required=True, help="Frozen shadow artifact directory")
    quarantine_audit.add_argument(
        "--snapshot-root",
        required=True,
        help="Root directory containing the immutable research snapshot",
    )
    quarantine_audit.add_argument(
        "--research-run",
        required=True,
        help="Research run directory containing verified job/cycle/quality/summary evidence",
    )
    quarantine_audit.add_argument(
        "--audit-ledger",
        required=True,
        help="Persistent tamper-evident sealed-quarantine audit ledger JSON",
    )
    quarantine_audit.add_argument(
        "--report",
        default=None,
        help="Optional atomic machine-readable quarantine audit JSON report",
    )

    deployment_review = subparsers.add_parser(
        "deployment-review",
        help=(
            "Assemble independently verified research, sealed-quarantine and forward-paper "
            "evidence for manual live review only"
        ),
    )
    deployment_review.add_argument("--artifact", required=True, help="Frozen shadow artifact directory")
    deployment_review.add_argument(
        "--snapshot-root",
        required=True,
        help="Root directory containing immutable snapshots referenced by the paper ledger",
    )
    deployment_review.add_argument(
        "--research-run",
        required=True,
        help="Research run directory containing job/cycle/quality/summary evidence",
    )
    deployment_review.add_argument(
        "--audit-ledger",
        required=True,
        help="Integrity-verified sealed-quarantine audit ledger JSON",
    )
    deployment_review.add_argument(
        "--report",
        default=None,
        help="Optional atomic machine-readable deployment review JSON report",
    )
    deployment_review.add_argument("--min-sessions", type=int, default=60)
    deployment_review.add_argument("--min-span-days", type=int, default=45)
    return parser


def resolve_shadow_provider(name: str):
    if name == "yahoo-bootstrap":
        return YahooBootstrapProvider()
    if name == "tiingo":
        token = os.environ.get("TIINGO_API_TOKEN", "").strip()
        if not token:
            raise ProviderError("TIINGO_API_TOKEN is required for --provider tiingo")
        return TiingoProvider(token=token)
    raise ValueError(f"unsupported shadow provider: {name}")


def _write_json_atomic(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def _validate_shadow_args(args: argparse.Namespace):
    if args.data_age_seconds is not None and args.data_age_seconds < 0.0:
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


def _snapshot_data_age_seconds(snapshot, *, now: datetime | None = None) -> float:
    if "retrieved_at" not in snapshot.bars.columns or snapshot.bars.empty:
        raise ValueError("verified snapshot is missing retrieval-time freshness evidence")
    retrieved = pd.to_datetime(snapshot.bars["retrieved_at"], utc=True, errors="coerce")
    if retrieved.isna().any():
        raise ValueError("verified snapshot contains invalid retrieval-time freshness evidence")

    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        raise ValueError("freshness reference time must be timezone-aware")
    reference = reference.astimezone(timezone.utc)
    newest = retrieved.max().to_pydatetime()
    oldest = retrieved.min().to_pydatetime()
    if newest > reference:
        raise ValueError("verified snapshot retrieval timestamp is in the future")
    return float((reference - oldest).total_seconds())


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


def _execute_shadow_snapshot(args: argparse.Namespace, snapshot, *, provider_refresh_verified: bool = False):
    auxiliary_names, auxiliary_store = _validate_shadow_args(args)
    if args.data_age_seconds is not None:
        data_age_seconds = float(args.data_age_seconds)
    elif provider_refresh_verified:
        data_age_seconds = 0.0
    else:
        data_age_seconds = _snapshot_data_age_seconds(snapshot)
    return run_shadow_step(
        snapshot.bars,
        artifact_dir=args.artifact,
        state_path=args.state,
        ledger_path=args.ledger,
        snapshot_fingerprint=snapshot.manifest.dataset_fingerprint,
        data_age_seconds=data_age_seconds,
        kill_switch=args.kill_switch,
        auxiliary_store=auxiliary_store,
        auxiliary_feature_names=auxiliary_names,
        auxiliary_max_age_days=args.auxiliary_max_age_days,
        auxiliary_min_coverage=args.auxiliary_min_coverage,
    )


def _print_shadow_result(result) -> None:
    print(f"shadow_strategy_id={result.strategy_id}")
    print(f"shadow_artifact_id={result.artifact_id}")
    print(f"processed_timestamp={result.processed_timestamp}")
    print(f"pending_signal_timestamp={result.pending_signal_timestamp}")
    print(f"target_count={sum(abs(float(value)) > 1e-12 for value in result.target_weights.values())}")
    if result.observation is None:
        print("paper_observation=idempotent" if result.idempotent_replay else "paper_observation=pending")
    else:
        print(f"paper_observation={result.observation.timestamp}")
        if hasattr(result.observation, "net_return"):
            print(f"paper_net_return={result.observation.net_return:.6f}")
        if hasattr(result.observation, "fill_rate"):
            print(f"paper_fill_rate={result.observation.fill_rate:.6f}")
        if hasattr(result.observation, "cost_rate"):
            print(f"paper_cost_rate={result.observation.cost_rate:.6f}")
    print("broker_execution=disabled")


def _run_shadow_snapshot(args: argparse.Namespace, snapshot) -> int:
    result = _execute_shadow_snapshot(args, snapshot)
    _print_shadow_result(result)
    return 0


def _paper_step_status(result) -> str:
    if result.idempotent_replay:
        return "idempotent"
    if result.observation is not None:
        return "settled"
    return "pending"


def _successful_cycle_report(args: argparse.Namespace, artifact, snapshot, result, *, snapshot_status: str) -> dict:
    if bool(result.broker_execution_available):
        raise ValueError("shadow result may not enable broker execution")
    return {
        "schema_version": 1,
        "status": "ok",
        "provider": args.provider,
        "strategy_id": result.strategy_id,
        "artifact_id": result.artifact_id,
        "research_cycle_id": artifact.research_cycle_id,
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_fingerprint": snapshot.manifest.dataset_fingerprint,
        "snapshot_status": snapshot_status,
        "paper_step_status": _paper_step_status(result),
        "processed_timestamp": result.processed_timestamp,
        "pending_signal_timestamp": result.pending_signal_timestamp,
        "paper_observation_timestamp": (
            None if result.observation is None else result.observation.timestamp
        ),
        "target_count": sum(abs(float(value)) > 1e-12 for value in result.target_weights.values()),
        "broker_execution_available": False,
    }


def _successful_status_report(report, provenance_report) -> dict:
    if bool(getattr(provenance_report, "broker_execution_available", False)):
        raise ValueError("paper provenance verification may not enable broker execution")
    return {
        "schema_version": 1,
        "status": "ok",
        "strategy_id": report.strategy_id,
        "sessions": int(report.sessions),
        "live_span_days": int(report.live_span_days),
        "paper_sharpe": float(report.metrics.get("sharpe", float("nan"))),
        "paper_cagr": float(report.metrics.get("cagr", float("nan"))),
        "paper_excess_cagr": float(report.excess_cagr),
        "paper_max_drawdown": float(report.metrics.get("max_drawdown", float("nan"))),
        "average_fill_rate": float(report.average_fill_rate),
        "evidence_score": float(report.evidence_score),
        "frozen_provenance_complete": bool(report.frozen_provenance_complete),
        "external_provenance_verified": bool(provenance_report.verified),
        "verified_snapshots": int(provenance_report.snapshots_verified),
        "snapshot_fingerprints": list(provenance_report.snapshot_fingerprints),
        "artifact_verified": bool(provenance_report.artifact_verified),
        "model_hash_verified": bool(provenance_report.model_hash_verified),
        "artifact_strategy_match": bool(provenance_report.artifact_strategy_match),
        "artifact_cycle_match": bool(provenance_report.artifact_cycle_match),
        "signal_snapshots_verified": int(provenance_report.signal_snapshots_verified),
        "realization_snapshots_verified": int(provenance_report.realization_snapshots_verified),
        "snapshot_timeline_verified": bool(provenance_report.snapshot_timeline_verified),
        "observations_verified": int(provenance_report.observations_verified),
        "provenance_reasons": list(provenance_report.reasons),
        "model_artifact_id": report.model_artifact_id,
        "research_cycle_id": report.research_cycle_id,
        "live_evidence_eligible": bool(report.live_eligible),
        "reasons": list(report.reasons),
        "broker_execution_available": False,
    }


def _successful_quarantine_audit_report(artifact, research, snapshot, result) -> dict:
    return {
        "schema_version": 1,
        "status": "ok",
        "audit_id": str(result.record.audit_id),
        "strategy_id": str(artifact.strategy_id),
        "artifact_id": str(artifact.artifact_id),
        "experiment_id": str(artifact.experiment_id),
        "research_cycle_id": str(research.research_cycle_id),
        "dataset_fingerprint": str(snapshot.manifest.dataset_fingerprint),
        "quarantine_id": str(result.manifest.quarantine_id),
        "quarantine_start": str(result.manifest.start),
        "score": float(result.record.score),
        "passed": bool(result.record.passed),
        "reasons": list(result.record.reasons),
        "broker_execution_available": False,
    }


def _successful_deployment_review_report(artifact, research, audit, paper_report, provenance_report, review) -> dict:
    if bool(getattr(review, "broker_execution_available", False)):
        raise ValueError("deployment review may not enable broker execution")
    return {
        "schema_version": 1,
        "status": "ok",
        "strategy_id": str(artifact.strategy_id),
        "artifact_id": str(artifact.artifact_id),
        "experiment_id": str(artifact.experiment_id),
        "research_cycle_id": str(artifact.research_cycle_id),
        "dataset_fingerprint": str(research.dataset_fingerprint),
        "quality_fingerprint": str(research.quality_fingerprint),
        "quarantine_start": research.quarantine_start,
        "audit_id": getattr(audit, "audit_id", None),
        "research_ready": bool(review.research_ready),
        "research_grade": bool(review.research_grade),
        "quarantine_audit_passed": bool(review.quarantine_audit_passed),
        "paper_live_evidence_eligible": bool(review.paper_live_evidence_eligible),
        "paper_frozen_provenance_complete": bool(review.paper_frozen_provenance_complete),
        "paper_external_provenance_verified": bool(review.paper_external_provenance_verified),
        "paper_sessions": int(paper_report.sessions),
        "paper_live_span_days": int(paper_report.live_span_days),
        "external_provenance_verified": bool(provenance_report.verified),
        "artifact_verified": bool(provenance_report.artifact_verified),
        "model_hash_verified": bool(provenance_report.model_hash_verified),
        "snapshot_timeline_verified": bool(provenance_report.snapshot_timeline_verified),
        "observations_verified": int(provenance_report.observations_verified),
        "hard_risk_engine_required": bool(review.hard_risk_engine_required),
        "eligible_for_manual_live_review": bool(review.eligible_for_manual_live_review),
        "reasons": list(review.reasons),
        "broker_execution_available": False,
    }


def _run_shadow_cycle(args: argparse.Namespace) -> int:
    artifact = load_frozen_shadow_artifact(args.artifact)
    if args.start is None:
        baseline = _select_latest_compatible_snapshot(args.snapshot_root, artifact.symbols)
        start = baseline.manifest.start
    else:
        start = args.start
    end = args.end or datetime.now(timezone.utc).date().isoformat()
    provider = resolve_shadow_provider(args.provider)
    root = Path(args.snapshot_root)
    existing_snapshot_ids = {
        child.name for child in root.iterdir() if child.is_dir()
    } if root.is_dir() else set()
    store = SnapshotStore(root)
    snapshot = download_market_snapshot(
        provider,
        artifact.symbols,
        start,
        end,
        store,
        provenance={
            "purpose": "frozen_forward_shadow",
            "shadow_artifact_id": artifact.artifact_id,
            "research_cycle_id": artifact.research_cycle_id,
        },
        reuse_identical=True,
    )
    snapshot_status = "reused" if snapshot.snapshot_id in existing_snapshot_ids else "new"
    result = _execute_shadow_snapshot(args, snapshot, provider_refresh_verified=True)
    if args.report:
        _write_json_atomic(
            args.report,
            _successful_cycle_report(
                args,
                artifact,
                snapshot,
                result,
                snapshot_status=snapshot_status,
            ),
        )
    print(f"refreshed_snapshot_id={snapshot.snapshot_id}")
    print(f"refreshed_snapshot_fingerprint={snapshot.manifest.dataset_fingerprint}")
    _print_shadow_result(result)
    return 0


def _run_locked_shadow_command(args: argparse.Namespace) -> int:
    if args.command == "step":
        snapshot = SnapshotStore(args.snapshot_root).load(args.snapshot_id)
        return _run_shadow_snapshot(args, snapshot)

    if args.command == "auto-step":
        artifact = load_frozen_shadow_artifact(args.artifact)
        snapshot = _select_latest_compatible_snapshot(args.snapshot_root, artifact.symbols)
        print(f"selected_snapshot_id={snapshot.snapshot_id}")
        print(f"selected_snapshot_fingerprint={snapshot.manifest.dataset_fingerprint}")
        return _run_shadow_snapshot(args, snapshot)

    if args.command == "cycle":
        return _run_shadow_cycle(args)

    raise ValueError(f"unsupported shadow command: {args.command}")


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

    if args.command in {"step", "auto-step", "cycle"}:
        lock_path = shadow_command_lock_path(args.state, getattr(args, "lock", None))
        with exclusive_shadow_command_lock(lock_path):
            return _run_locked_shadow_command(args)

    if args.command == "quarantine-audit":
        artifact = load_frozen_shadow_artifact(args.artifact)
        research = verify_research_evidence_bundle(
            args.research_run,
            artifact_manifest=artifact,
        )
        if research.quarantine_start is None:
            raise ValueError("quarantine audit requires sealed research quarantine boundary")
        if not bool(research.research_ready):
            raise ValueError("quarantine audit requires research-ready frozen champion")
        if research.data_grade is not DataGrade.RESEARCH_GRADE:
            raise ValueError("quarantine audit requires verified RESEARCH_GRADE data")

        snapshot = SnapshotStore(args.snapshot_root).find_verified_by_fingerprint(
            research.dataset_fingerprint
        )
        if snapshot is None:
            raise ValueError("verified research snapshot not found for quarantine audit")
        artifact_symbols = tuple(sorted(str(value).upper() for value in artifact.symbols))
        if tuple(snapshot.manifest.symbols) != artifact_symbols:
            raise ValueError("research snapshot universe does not match frozen artifact universe")

        spec = FrozenStrategySpec(
            strategy_id=str(artifact.strategy_id),
            experiment_id=str(artifact.experiment_id),
            horizon=int(artifact.horizon),
            model_name=str(artifact.model_name),
            model_params=dict(artifact.model_params),
            seed=int(artifact.seed),
            top_fraction=float(artifact.top_fraction),
            weighting=str(artifact.weighting),
        )
        quarantine_config = QuarantineConfig(
            start=research.quarantine_start,
            min_development_periods=1,
            min_quarantine_periods=1,
        )
        result = run_single_quarantine_audit(
            snapshot.bars,
            quarantine_config,
            spec,
            ledger_path=args.audit_ledger,
            research_cycle_id=research.research_cycle_id,
        )
        if args.report:
            _write_json_atomic(
                args.report,
                _successful_quarantine_audit_report(artifact, research, snapshot, result),
            )
        print(f"quarantine_audit_id={result.record.audit_id}")
        print(f"strategy_id={artifact.strategy_id}")
        print(f"experiment_id={artifact.experiment_id}")
        print(f"research_cycle_id={research.research_cycle_id}")
        print(f"quarantine_id={result.manifest.quarantine_id}")
        print(f"quarantine_audit_passed={'yes' if result.record.passed else 'no'}")
        print("reasons=" + (",".join(result.record.reasons) if result.record.reasons else "none"))
        print("broker_execution=disabled")
        return 0

    if args.command == "status":
        records = ledger.records(strategy_id=args.strategy_id)
        if not records:
            raise ValueError("no paper observations found for strategy")
        criteria = PaperArenaCriteria(
            min_sessions=args.min_sessions,
            min_live_span_days=args.min_span_days,
            require_frozen_provenance=True,
        )
        report = evaluate_paper_track(records, criteria=criteria)
        provenance_report = verify_frozen_paper_provenance(
            records,
            artifact_dir=args.artifact,
            snapshot_root=args.snapshot_root,
        )
        if args.report:
            _write_json_atomic(
                args.report,
                _successful_status_report(report, provenance_report),
            )
        print(f"strategy_id={report.strategy_id}")
        print(f"sessions={report.sessions}")
        print(f"live_span_days={report.live_span_days}")
        print(f"paper_sharpe={report.metrics.get('sharpe', float('nan')):.6f}")
        print(f"paper_cagr={report.metrics.get('cagr', float('nan')):.6f}")
        print(f"paper_excess_cagr={report.excess_cagr:.6f}")
        print(f"paper_max_drawdown={report.metrics.get('max_drawdown', float('nan')):.6f}")
        print(f"average_fill_rate={report.average_fill_rate:.6f}")
        print(f"evidence_score={report.evidence_score:.6f}")
        print(
            f"frozen_provenance_complete={'yes' if report.frozen_provenance_complete else 'no'}"
        )
        print(
            f"external_provenance_verified={'yes' if provenance_report.verified else 'no'}"
        )
        print(f"verified_snapshots={provenance_report.snapshots_verified}")
        if report.model_artifact_id is not None:
            print(f"model_artifact_id={report.model_artifact_id}")
        if report.research_cycle_id is not None:
            print(f"research_cycle_id={report.research_cycle_id}")
        print(f"live_evidence_eligible={'yes' if report.live_eligible else 'no'}")
        print("reasons=" + (",".join(report.reasons) if report.reasons else "none"))
        print("broker_execution=disabled")
        return 0

    if args.command == "deployment-review":
        artifact = load_frozen_shadow_artifact(args.artifact)
        research = verify_research_evidence_bundle(
            args.research_run,
            artifact_manifest=artifact,
        )
        if research.quarantine_start is None:
            raise ValueError("deployment review requires sealed research quarantine boundary")
        audit = load_verified_quarantine_audit_record(
            args.audit_ledger,
            quarantine_start=research.quarantine_start,
            strategy_id=artifact.strategy_id,
            research_cycle_id=research.research_cycle_id,
        )
        records = ledger.records(strategy_id=artifact.strategy_id)
        if not records:
            raise ValueError("no paper observations found for frozen strategy")
        criteria = PaperArenaCriteria(
            min_sessions=args.min_sessions,
            min_live_span_days=args.min_span_days,
            require_frozen_provenance=True,
        )
        paper_report = evaluate_paper_track(records, criteria=criteria)
        provenance_report = verify_frozen_paper_provenance(
            records,
            artifact_dir=args.artifact,
            snapshot_root=args.snapshot_root,
        )
        review = evaluate_bound_deployment_review(
            research_evidence=research,
            artifact_manifest=artifact,
            quarantine_audit_record=audit,
            paper_report=paper_report,
            paper_provenance_report=provenance_report,
        )
        if args.report:
            _write_json_atomic(
                args.report,
                _successful_deployment_review_report(
                    artifact,
                    research,
                    audit,
                    paper_report,
                    provenance_report,
                    review,
                ),
            )
        print(f"strategy_id={artifact.strategy_id}")
        print(f"artifact_id={artifact.artifact_id}")
        print(f"experiment_id={artifact.experiment_id}")
        print(f"research_cycle_id={artifact.research_cycle_id}")
        print(
            "eligible_for_manual_live_review="
            f"{'yes' if review.eligible_for_manual_live_review else 'no'}"
        )
        print("reasons=" + (",".join(review.reasons) if review.reasons else "none"))
        print("hard_risk_engine_required=yes")
        print("broker_execution=disabled")
        return 0

    raise ValueError(f"unsupported command: {args.command}")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_from_args(args)
    except (ProviderError, DownloadError, ValueError) as exc:
        if args.command == "cycle" and getattr(args, "report", None):
            _write_json_atomic(
                args.report,
                {
                    "schema_version": 1,
                    "status": "error",
                    "provider": args.provider,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "broker_execution_available": False,
                },
            )
        if args.command == "quarantine-audit" and getattr(args, "report", None):
            _write_json_atomic(
                args.report,
                {
                    "schema_version": 1,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "broker_execution_available": False,
                },
            )
        if args.command == "deployment-review" and getattr(args, "report", None):
            _write_json_atomic(
                args.report,
                {
                    "schema_version": 1,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "eligible_for_manual_live_review": False,
                    "broker_execution_available": False,
                },
            )
        parser.error(str(exc))
    return 2
