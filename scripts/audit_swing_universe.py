"""Offline historical equity-universe audit; NEVER trades or makes alpha claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stockbot.swing.data_quality import XSTOProvenance
from stockbot.swing.universe import audit_instrument_history, parse_universe_records


def main() -> int:
    p = argparse.ArgumentParser(description="XSTO lifecycle audit ONLY, not backtesting")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--universe-json", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, help="Optional audit receipt JSON")
    args = p.parse_args()
    for file in (args.csv, args.universe_json, args.manifest):
        if not file.is_file():
            p.error(f"missing input: {file}")
    manifest = XSTOProvenance.from_dict(json.loads(
        args.manifest.read_text(encoding="utf-8")
    ))
    records = parse_universe_records(json.loads(
        args.universe_json.read_text(encoding="utf-8")
    ))
    receipt = audit_instrument_history(
        pd.read_csv(args.csv), records, collected_at_utc=manifest.collected_at_utc
    ).to_dict()
    receipt["order_execution_permitted"] = False
    receipt["source_usage_rights_verified"] = False
    receipt["source_provider"] = manifest.provider
    receipt["warning"] = (
        "STRUCTURAL ONLY. Self-attested universe is not proof that delisted, "
        "suspended or renamed stocks are all represented. Uneven lifecycle "
        "datasets CANNOT currently be replayed by the fixed-panel trading engine."
    )
    payload = json.dumps(receipt, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
