from __future__ import annotations

import json

from stockbot.runtime.readiness import paper_readiness


def main() -> int:
    report = paper_readiness(466.0)
    print(
        json.dumps(
            {
                "paper_shadow_ready": report.paper_shadow_ready,
                "live_ready": report.live_ready,
                "checks": report.checks,
            },
            sort_keys=True,
        )
    )
    return 0 if report.paper_shadow_ready and not report.live_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
