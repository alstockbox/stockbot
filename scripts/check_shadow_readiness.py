from __future__ import annotations

import json

from stockbot.runtime.shadow_readiness import shadow_readiness


def main() -> int:
    report = shadow_readiness(466.0)
    print(
        json.dumps(
            {
                "shadow_ready": report.ready,
                "live_ready": report.live_ready,
                "checks": report.checks,
            },
            sort_keys=True,
        )
    )
    if report.ready and not report.live_ready:
        print("SHADOW_READINESS_OK")
        return 0
    print("SHADOW_READINESS_FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
