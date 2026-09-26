from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import time

from stockbot.brokers.snapshot_feed import SnapshotFileProvider
from stockbot.runtime.live_planner import LiveShadowPlanner
from stockbot.runtime.shadow_runtime import ShadowTradingRuntime


def _write_demo_snapshot(path: Path, symbols: list[str]) -> None:
    now = datetime.now(timezone.utc)
    payload = {
        "account": {
            "balance": 466.0,
            "equity": 466.0,
            "free_margin": 466.0,
            "currency": "USD",
            "timestamp": now.isoformat(),
        },
        "symbols": {},
    }
    for symbol in symbols:
        price = 100.0
        bars = []
        start = now - timedelta(minutes=5 * 260)
        for index in range(250):
            open_price = price
            price *= 1.001
            bars.append(
                {
                    "timestamp": (start + timedelta(minutes=5 * index)).isoformat(),
                    "open": open_price,
                    "high": price * 1.001,
                    "low": open_price * 0.999,
                    "close": price,
                    "volume": 1000.0 + index,
                }
            )
        payload["symbols"][symbol] = {
            "quote": {
                "bid": price * 0.9999,
                "ask": price * 1.0001,
                "timestamp": now.isoformat(),
            },
            "spec": {
                "tick_size": 0.00001,
                "tick_value": 1.0,
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
                "trade_enabled": True,
            },
            "bars": bars,
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate causal opportunities from closed bars and execute them "
            "only in StockBot shadow mode."
        )
    )
    parser.add_argument("--snapshot-file", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--symbols", default="EURUSD")
    parser.add_argument("--journal", type=Path, default=Path("var/live_shadow_journal.jsonl"))
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    return parser


def _cycle(
    provider: SnapshotFileProvider,
    runtime: ShadowTradingRuntime,
    planner: LiveShadowPlanner,
    symbols: list[str],
) -> dict:
    now = datetime.now(timezone.utc)
    plans = []
    analyses = []
    errors = []

    for symbol in symbols:
        try:
            result = planner.build(symbol)
            analyses.append(
                {
                    "symbol": symbol,
                    "regime": result.analysis.regime.value,
                    "signal": result.analysis.combined_signal,
                    "expected_return": result.analysis.expected_return,
                    "downside_risk": result.analysis.downside_risk,
                    "uncertainty": result.analysis.uncertainty,
                    "sample_count": result.analysis.sample_count,
                    "confidence": result.analysis.confidence,
                    "eligible": result.analysis.eligible,
                }
            )
            if result.plan is not None:
                plans.append(result.plan)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": type(exc).__name__})

    execution = runtime.run_cycle(
        plans,
        timestamp=now,
        regime="multi_asset_live_shadow",
    )

    return {
        "analyses": analyses,
        "errors": errors,
        "plans": len(plans),
        "accepted": execution.accepted,
        "rejected": execution.rejected,
        "open_shadow_positions": len(runtime.executor.positions),
        "live_orders_possible": False,
    }


def _run(
    snapshot_file: Path,
    args: argparse.Namespace,
    symbols: list[str],
) -> int:
    provider = SnapshotFileProvider(snapshot_file)
    runtime = ShadowTradingRuntime(provider, args.journal)
    planner = LiveShadowPlanner(provider)

    while True:
        payload = _cycle(provider, runtime, planner, symbols)
        print(json.dumps(payload, sort_keys=True))
        if not args.loop:
            return 0
        try:
            time.sleep(max(1.0, args.interval_seconds))
        except KeyboardInterrupt:
            print("LIVE_SHADOW_LOOP_STOPPED")
            return 0


def main() -> int:
    args = _parser().parse_args()
    symbols = [symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()]
    if not symbols:
        print("NO_SYMBOLS_CONFIGURED")
        return 2

    if args.demo:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "stockbot_snapshot.json"
            _write_demo_snapshot(snapshot, symbols)
            return _run(snapshot, args, symbols)

    if args.snapshot_file is None:
        print(
            "LIVE_SHADOW_FEED_NOT_CONFIGURED: use --demo or "
            "--snapshot-file PATH. Broker writes are not available."
        )
        return 2

    return _run(args.snapshot_file, args, symbols)


if __name__ == "__main__":
    raise SystemExit(main())
