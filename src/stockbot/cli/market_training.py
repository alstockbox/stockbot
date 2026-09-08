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
from stockbot.research.market_training import run_snapshot_factory, train_snapshot
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
        config = ResearchFactoryConfig(
            horizons=_parse_horizons(args.factory_horizons),
            population=ModelPopulationConfig(max_candidates=args.factory_candidates),
            max_workers=args.factory_workers,
        )
        report = run_snapshot_factory(
            snapshot,
            config=config,
            memory_path=args.factory_memory,
        )
        print("research_factory:")
        print(f"  experiments_run={report.experiments_run}")
        print(f"  candidates_passed={report.candidates_passed}")
        for index, candidate in enumerate(report.candidates[:20], start=1):
            gate = "pass" if candidate.gate.passed else "reject"
            print(
                f"  {index}. h={candidate.horizon} {candidate.model_name} "
                f"factory_score={candidate.factory_score:.6f} base_score={candidate.base_score:.6f} "
                f"oos={candidate.oos_coverage:.3f} gate={gate}"
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
