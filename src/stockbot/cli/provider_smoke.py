from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from stockbot.data.providers.http import ProviderError
from stockbot.data.providers.tiingo import TiingoProvider
from stockbot.data.providers.yahoo_bootstrap import YahooBootstrapProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test external StockBot market-data providers. No broker execution is available."
    )
    parser.add_argument(
        "--provider",
        choices=("all", "yahoo-bootstrap", "tiingo"),
        default="all",
    )
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--report", default=None)
    return parser


def _write_json_atomic(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def _smoke_provider(provider, *, symbol: str, start: str, end: str) -> dict:
    frame = provider.fetch_bars(symbol, start, end)
    if frame.empty:
        raise ProviderError(f"{provider.name} returned no rows")
    if "timestamp" not in frame.columns:
        raise ProviderError(f"{provider.name} returned no timestamp column")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ProviderError(f"{provider.name} returned invalid timestamps")
    grade = provider.grade
    return {
        "status": "ok",
        "grade": grade.value,
        "rows": int(len(frame)),
        "first_timestamp": timestamps.min().isoformat(),
        "last_timestamp": timestamps.max().isoformat(),
    }


def run_from_args(args: argparse.Namespace) -> int:
    symbol = str(args.symbol).strip().upper()
    if not symbol:
        raise ValueError("symbol is required")

    providers: dict[str, dict] = {}
    if args.provider in {"all", "yahoo-bootstrap"}:
        providers["yahoo-bootstrap"] = _smoke_provider(
            YahooBootstrapProvider(),
            symbol=symbol,
            start=args.start,
            end=args.end,
        )

    if args.provider in {"all", "tiingo"}:
        token = os.environ.get("TIINGO_API_TOKEN", "").strip()
        if not token:
            if args.provider == "tiingo":
                raise ProviderError("TIINGO_API_TOKEN is required for Tiingo provider smoke")
            providers["tiingo"] = {
                "status": "skipped",
                "reason": "TIINGO_API_TOKEN is not configured",
            }
        else:
            providers["tiingo"] = _smoke_provider(
                TiingoProvider(token=token),
                symbol=symbol,
                start=args.start,
                end=args.end,
            )

    report = {
        "schema_version": 1,
        "status": "ok",
        "symbol": symbol,
        "start": str(args.start),
        "end": str(args.end),
        "providers": providers,
        "broker_execution_available": False,
    }
    if args.report:
        _write_json_atomic(args.report, report)

    for name, result in providers.items():
        print(f"provider={name} status={result['status']}")
    print("broker_execution=disabled")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_from_args(args)
    except (ProviderError, ValueError) as exc:
        if args.report:
            _write_json_atomic(
                args.report,
                {
                    "schema_version": 1,
                    "status": "error",
                    "symbol": str(args.symbol).strip().upper(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "broker_execution_available": False,
                },
            )
        print(f"provider_smoke_error={exc}")
        print("broker_execution=disabled")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
