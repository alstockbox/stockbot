from __future__ import annotations

import argparse
from pathlib import Path

from stockbot.data.providers.http import ProviderError
from stockbot.data.providers.riksbank_macro import (
    DEFAULT_SERIES,
    RiksbankMonetaryPolicyProvider,
    RiksbankSeries,
)
from stockbot.data.research_inputs import write_point_in_time_feature_store


_SERIES_BY_FEATURE = {item.feature_name: item for item in DEFAULT_SERIES}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fingerprinted StockBot point-in-time macro feature store from "
            "Sveriges Riksbank Monetary Policy Data vintages."
        )
    )
    parser.add_argument("--output", required=True, help="Output research-input JSON path")
    parser.add_argument(
        "--series",
        default=",".join(_SERIES_BY_FEATURE),
        help="Comma-separated StockBot Riksbank feature aliases",
    )
    parser.add_argument(
        "--policy-round",
        default=None,
        help="Optional single Riksbank policy round YYYY:N; omit to collect historical vintages",
    )
    parser.add_argument("--start-year", type=int, default=2020, help="First vintage year when collecting history")
    parser.add_argument("--end-year", type=int, default=None, help="Optional final vintage year")
    return parser


def _parse_series(value: str) -> tuple[RiksbankSeries, ...]:
    names = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not names:
        raise ValueError("at least one Riksbank feature alias is required")
    if len(set(names)) != len(names):
        raise ValueError("Riksbank feature aliases must be unique")
    unknown = sorted(set(names).difference(_SERIES_BY_FEATURE))
    if unknown:
        raise ValueError(
            "unknown Riksbank feature aliases: "
            + ", ".join(unknown)
            + "; available: "
            + ", ".join(sorted(_SERIES_BY_FEATURE))
        )
    return tuple(_SERIES_BY_FEATURE[name] for name in names)


def run_from_args(args: argparse.Namespace, *, provider=None) -> int:
    selected = _parse_series(args.series)
    if args.start_year < 2020:
        raise ValueError("Riksbank forecast/outcome vintages are supported from 2020")
    if args.end_year is not None and args.end_year < args.start_year:
        raise ValueError("end-year cannot precede start-year")

    source = provider or RiksbankMonetaryPolicyProvider()
    if args.policy_round:
        store = source.build_store(selected, policy_round=args.policy_round)
        mode = f"round:{args.policy_round}"
    else:
        store = source.build_vintage_store(
            selected,
            start_year=args.start_year,
            end_year=args.end_year,
        )
        end = "latest" if args.end_year is None else str(args.end_year)
        mode = f"vintages:{args.start_year}-{end}"

    output = Path(args.output)
    write_point_in_time_feature_store(output, store)
    print(f"output={output}")
    print(f"source={store.manifest.source}")
    print(f"mode={mode}")
    print(f"features={','.join(store.feature_names)}")
    print(f"observations={len(store.observations)}")
    print(f"fingerprint={store.fingerprint}")
    print("point_in_time=true")
    print("revision_aware=true")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_from_args(args)
    except (ProviderError, ValueError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
