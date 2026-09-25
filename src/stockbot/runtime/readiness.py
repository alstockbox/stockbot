from __future__ import annotations

from dataclasses import dataclass

from stockbot.execution.safeguard import SafeguardConfig
from stockbot.execution.smoke import run_smoke_test


@dataclass(frozen=True)
class ReadinessReport:
    paper_shadow_ready: bool
    live_ready: bool
    checks: dict[str, bool]


def paper_readiness(equity_usd: float = 466.0) -> ReadinessReport:
    smoke = run_smoke_test(equity_usd)
    config = SafeguardConfig()
    checks = {
        **{f"smoke:{name}": passed for name, passed in smoke.items()},
        "live_disabled_by_default": not config.live_execution_enabled,
        "positive_order_risk_limit": 0.0 < config.max_order_risk_pct < 1.0,
        "positive_total_risk_limit": 0.0 < config.max_total_open_risk_pct < 1.0,
        "drawdown_circuit_breaker_configured": 0.0 < config.max_drawdown_pct < 1.0,
        "daily_loss_circuit_breaker_configured": 0.0 < config.max_daily_loss_pct < 1.0,
    }
    return ReadinessReport(
        paper_shadow_ready=all(checks.values()),
        live_ready=False,
        checks=checks,
    )
