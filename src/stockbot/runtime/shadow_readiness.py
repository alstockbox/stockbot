from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

from stockbot.brokers.mt5_readonly import MT5ReadOnlyAdapter
from stockbot.data.live_models import AccountSnapshot, MarketQuote, SymbolSpec
from stockbot.data.live_validation import (
    LiveValidationConfig,
    validate_account,
    validate_quote,
    validate_symbol,
)
from stockbot.execution.safeguard import SafeguardConfig
from stockbot.execution.shadow import ShadowExecutor
from stockbot.execution.shadow_state import ShadowPortfolioState, ShadowStateStore


@dataclass(frozen=True)
class ShadowReadinessReport:
    ready: bool
    live_ready: bool
    checks: dict[str, bool]


def shadow_readiness(equity_usd: float = 466.0) -> ShadowReadinessReport:
    now = datetime.now(timezone.utc)
    validation = LiveValidationConfig()
    safe_quote = MarketQuote("EURUSD", 1.1000, 1.1002, now, "readiness")
    stale_quote = MarketQuote(
        "EURUSD",
        1.1000,
        1.1002,
        now - timedelta(seconds=validation.max_quote_age_seconds + 1.0),
        "readiness",
    )
    spec = SymbolSpec("EURUSD", 0.00001, 1.0, 0.01, 100.0, 0.01, True)
    account = AccountSnapshot(equity_usd, equity_usd, equity_usd, now, "USD")
    executor = ShadowExecutor()
    position = executor.open_long(
        decision_id="readiness",
        quote=safe_quote,
        notional_usd=min(100.0, equity_usd * 0.25),
        estimated_risk_usd=min(1.0, equity_usd * 0.003),
    )

    with tempfile.TemporaryDirectory() as tmp:
        store = ShadowStateStore(Path(tmp) / "shadow_state.json")
        store.save(
            ShadowPortfolioState(
                base_equity_usd=equity_usd,
                realized_pnl_usd=0.0,
                equity_peak_usd=equity_usd,
                positions=(position,),
                updated_at=now,
            )
        )
        restored = store.load()
        persistence_roundtrip = (
            restored is not None
            and len(restored.positions) == 1
            and restored.positions[0].decision_id == "readiness"
        )

    checks = {
        "live_execution_disabled": not SafeguardConfig().live_execution_enabled,
        "mt5_adapter_has_no_order_send": not hasattr(MT5ReadOnlyAdapter, "order_send"),
        "mt5_adapter_has_no_order_check": not hasattr(MT5ReadOnlyAdapter, "order_check"),
        "safe_quote_valid": validate_quote(safe_quote, now=now, config=validation).ok,
        "stale_quote_rejected": not validate_quote(
            stale_quote,
            now=now,
            config=validation,
        ).ok,
        "account_valid": validate_account(account, now=now, config=validation).ok,
        "symbol_valid": validate_symbol(spec).ok,
        "shadow_position_opened_without_broker_write": position.decision_id == "readiness",
        "shadow_state_persistence_roundtrip": persistence_roundtrip,
    }
    return ShadowReadinessReport(
        ready=all(checks.values()),
        live_ready=False,
        checks=checks,
    )
