from __future__ import annotations

import argparse
import os
from pathlib import Path

from stockbot.data.download import DownloadError, download_market_snapshot
from stockbot.data.providers.http import ProviderError
from stockbot.data.providers.tiingo import TiingoProvider
from stockbot.data.providers.yahoo_bootstrap import YahooBootstrapProvider
from stockbot.data.snapshots import SnapshotStore
from stockbot.research.factory import ResearchFactoryConfig
from stockbot.research.jobs import make_job_manifest, make_run_summary, write_json_record
from stockbot.research.market_training import (
    run_snapshot_factory,
    run_snapshot_regime_specialists,
    train_snapshot,
)
from stockbot.research.population import ModelPopulationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download real EOD market data into an immutable StockBot snapshot and "
            "optionally run the V1 Alpha Arena or the V2 multi-horizon research factory."
        )
    )
    parser.add_argument("--provider", required=True, choices=("tiingo", "yahoo-bootstrap"))
    parser.add_argument("--symbols", required=True, help="Comma-separated tickers, e.g. AAPL,MSFT,SPY")
    parser.add_argument("--start", required=True, help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--snapshot-root", default="snapshots", help="Directory for immutable snapshots")
    parser.add_argument("--train", action="store_true", help="Run the existing V1 Alpha Arena after download")
    parser.add_argument("--horizon", type=int, default=5, help="Forward-return horizon used by V1 training")
    parser.add_argument("--factory", action="store_true", help="Run the V2 autonomous research factory")
    parser.add_argument(
        "--factory-horizons",
        default="1,5,20",
        help="Comma-separated forward-return horizons for V2, e.g. 1,5,20",
    )
    parser.add_argument(
        "--factory-candidates",
        type=int,
        default=160,
        help="Maximum ML challenger configurations explored per horizon",
    )
    parser.add_argument(
        "--factory-workers",
        type=int,
        default=4,
        help="Parallel challenger workers used by V2",
    )
    parser.add_argument(
        "--factory-memory",
        default="research_memory/experiments.jsonl",
        help="Append-only research-memory JSONL path",
    )
    parser.add_argument(
        "--factory-run-dir",
        default=None,
        help="Optional directory for scheduler-friendly job manifest and run summary JSON",
    )
    parser.add_argument(
        "--factory-regime-specialists",
        action="store_true",
        help="After the main factory, train diagnostic bull/bear/chop specialists on the same snapshot",
    )
    parser.add_argument(
        "--factory-specialists-top-k",
        type=int,
        default=2,
        help="Generalist candidates per horizon admitted to regime-specialist research",
    )
    return parser


def resolve_provider(name: str):
    if name == "yahoo-bootstrap":
        return YahooBootstrapProvider()
    if name == "tiingo":
        token = os.environ.get("TIINGO_API_TOKEN", "").strip()
        if not token:
            raise ProviderError("TIINGO_API_TOKEN is required for --provider tiingo")
        return TiingoProvider(token=token)
    raise ValueError(f"unsupported provider: {name}")


def _parse_symbols(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _parse_horizons(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("factory horizons must be comma-separated integers") from exc
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("factory horizons must contain positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("factory horizons must be unique")
    return horizons


def run_from_args(args: argparse.Namespace) -> int:
    provider = resolve_provider(args.provider)
    symbols = _parse_symbols(args.symbols)
    store = SnapshotStore(Path(args.snapshot_root))
    snapshot = download_market_snapshot(provider, symbols, args.start, args.end, store)
    manifest = snapshot.manifest
    print(f"snapshot_id={snapshot.snapshot_id}")
    print(f"provider={manifest.provider}")
    print(f"grade={manifest.grade.value}")
    print(f"symbols={','.join(manifest.symbols)}")
    print(f"rows={manifest.row_count}")
    print(f"fingerprint={manifest.dataset_fingerprint}")
    print(f"path={snapshot.path}")

    if args.train:
        run = train_snapshot(snapshot, horizon=args.horizon)
        print("alpha_arena:")
        for index, result in enumerate(run.leaderboard, start=1):
            print(
                f"  {index}. {result.name} score={result.score:.6f} "
                f"oos={result.oos_coverage:.3f} drawdown={result.metrics.get('max_drawdown', float('nan')):.4f}"
            )
        champion = run.champion_candidate.name if run.champion_candidate is not None else "none"
        print(f"champion_candidate={champion}")

    if args.factory:
        horizons = _parse_horizons(args.factory_horizons)
        if args.factory_specialists_top_k <= 0:
            raise ValueError("factory specialists top-k must be positive")
        config = ResearchFactoryConfig(
            horizons=horizons,
            population=ModelPopulationConfig(max_candidates=args.factory_candidates),
            max_workers=args.factory_workers,
        )
        job_manifest = make_job_manifest(
            snapshot_id=snapshot.snapshot_id,
            dataset_fingerprint=manifest.dataset_fingerprint,
            horizons=horizons,
            max_candidates=args.factory_candidates,
            max_workers=args.factory_workers,
            memory_path=args.factory_memory,
        )
        run_dir = None
        if args.factory_run_dir:
            run_dir = Path(args.factory_run_dir) / job_manifest.job_id
            write_json_record(run_dir / "job.json", job_manifest)

        report = run_snapshot_factory(
            snapshot,
            config=config,
            memory_path=args.factory_memory,
        )
        specialist_diagnostics = None
        if args.factory_regime_specialists:
            specialist_diagnostics = run_snapshot_regime_specialists(
                snapshot,
                report,
                top_k_per_horizon=args.factory_specialists_top_k,
            )

        if run_dir is not None:
            write_json_record(
                run_dir / "summary.json",
                make_run_summary(job_manifest.job_id, report, specialist_diagnostics),
            )

        print("research_factory:")
        print(f"  job_id={job_manifest.job_id}")
        print(f"  experiments_run={report.experiments_run}")
        print(f"  promotion_candidates={report.candidates_passed}")
        print(f"  holdout_evaluated={report.holdout_evaluated}")
        print(f"  holdout_start={report.holdout_start}")
        for index, candidate in enumerate(report.candidates[:20], start=1):
            gate = "pass" if candidate.gate.passed else "reject"
            holdout = (
                "not_tested"
                if candidate.holdout_report is None
                else ("pass" if candidate.holdout_report.passed else "reject")
            )
            q_value = float("nan") if candidate.discovery is None else candidate.discovery.q_value
            regime_score = float("nan") if candidate.regime_report is None else candidate.regime_report.score
            drift = "unknown" if candidate.drift_report is None else ("degraded" if candidate.drift_report.degraded else "stable")
            readiness = (
                "unknown"
                if candidate.paper_readiness is None
                else ("ready" if candidate.paper_readiness.ready else "not_ready")
            )
            print(
                f"  {index}. h={candidate.horizon} {candidate.model_name} "
                f"research={candidate.factory_score:.6f} selection={candidate.selection_score:.6f} "
                f"promotion={candidate.promotion_score:.6f} oos={candidate.oos_coverage:.3f} "
                f"stress={candidate.stress_score:.3f} regime={regime_score:.3f} "
                f"q={q_value:.4f} drift={drift} readiness={readiness} gate={gate} holdout={holdout}"
            )
        if report.ensemble_report is not None:
            ensemble = report.ensemble_report
            print(
                f"horizon_ensemble_score={ensemble.score:.6f} "
                f"sharpe={ensemble.metrics.get('sharpe', float('nan')):.3f} "
                f"stress={ensemble.stress_report.score:.3f}"
            )
            print(
                "horizon_ensemble_weights="
                + ",".join(f"{key}:{weight:.4f}" for key, weight in ensemble.member_weights.items())
            )
        if specialist_diagnostics is not None:
            print(f"regime_specialist_candidate_pairs={specialist_diagnostics.candidate_pairs_tested}")
            for regime_name, specialist in specialist_diagnostics.specialists.items():
                print(
                    f"regime_specialist={regime_name} h={specialist.horizon} "
                    f"model={specialist.model_config.name} score={specialist.score:.6f} "
                    f"oos={specialist.oos_coverage:.3f} stress={specialist.stress_report.score:.3f}"
                )
            if specialist_diagnostics.router_report is not None:
                router = specialist_diagnostics.router_report
                print(
                    f"regime_router_score={router.score:.6f} "
                    f"sharpe={router.metrics.get('sharpe', float('nan')):.3f} "
                    f"stress={router.stress_report.score:.3f}"
                )
        champion = report.champion_candidate
        if champion is None:
            print("factory_champion_candidate=none")
        else:
            print(
                "factory_champion_candidate="
                f"h{champion.horizon}:{champion.model_name}:{champion.experiment_id}"
            )
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_from_args(args)
    except (ProviderError, DownloadError, ValueError) as exc:
        parser.error(str(exc))
    return 2
