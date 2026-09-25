from __future__ import annotations

import json
import sys

from stockbot.execution.smoke import run_smoke_test


def main() -> int:
    results = run_smoke_test(466.0)
    print(json.dumps(results, sort_keys=True))
    failed = [name for name, passed in results.items() if not passed]
    if failed:
        print("SAFEGUARD_SMOKE_FAILED:", ", ".join(failed), file=sys.stderr)
        return 1
    print("SAFEGUARD_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
