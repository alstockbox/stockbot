from __future__ import annotations

import argparse
from pathlib import Path

from stockbot.data.bolagsverket_research_input import build_store_from_manifest
from stockbot.data.research_inputs import write_point_in_time_feature_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a fingerprinted StockBot point-in-time fundamental store from Bolagsverket filings."
    )
    parser.add_argument("--manifest", required=True, help="Path to Bolagsverket filing manifest JSON")
    parser.add_argument("--output", required=True, help="Output research-input JSON artifact")
    parser.add_argument(
        "--include-derived",
        action="store_true",
        help="Add causal filing-vintage derived fundamental features",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    store = build_store_from_manifest(
        Path(args.manifest),
        include_derived=bool(args.include_derived),
    )
    output = Path(args.output)
    write_point_in_time_feature_store(output, store)
    print(
        "bolagsverket_research_input="
        f"{output} fingerprint={store.fingerprint} observations={len(store.observations)} "
        f"features={len(store.feature_names)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
