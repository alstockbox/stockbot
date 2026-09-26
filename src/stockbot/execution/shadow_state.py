from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

from stockbot.execution.shadow import ShadowPosition, ShadowSide


@dataclass(frozen=True)
class ShadowPortfolioState:
    base_equity_usd: float
    realized_pnl_usd: float
    equity_peak_usd: float
    positions: tuple[ShadowPosition, ...]
    updated_at: datetime
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.base_equity_usd) or self.base_equity_usd <= 0.0:
            raise ValueError("base_equity_usd must be positive and finite")
        if not math.isfinite(self.realized_pnl_usd):
            raise ValueError("realized_pnl_usd must be finite")
        if not math.isfinite(self.equity_peak_usd) or self.equity_peak_usd <= 0.0:
            raise ValueError("equity_peak_usd must be positive and finite")
        if self.updated_at.tzinfo is None:
            raise ValueError("updated_at must be timezone-aware")
        ids = [position.decision_id for position in self.positions]
        if len(ids) != len(set(ids)):
            raise ValueError("shadow positions must have unique decision_id values")

    def current_equity(self, unrealized_pnl_usd: float = 0.0) -> float:
        return float(
            self.base_equity_usd
            + self.realized_pnl_usd
            + float(unrealized_pnl_usd)
        )


class ShadowStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> ShadowPortfolioState | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("SHADOW_STATE_INVALID") from exc
        if int(payload.get("schema_version", 0)) != 1:
            raise RuntimeError("SHADOW_STATE_UNSUPPORTED_VERSION")

        positions = tuple(
            ShadowPosition(
                decision_id=row["decision_id"],
                symbol=row["symbol"],
                side=ShadowSide(row["side"]),
                notional_usd=float(row["notional_usd"]),
                estimated_risk_usd=float(row["estimated_risk_usd"]),
                quantity=float(row["quantity"]),
                entry_price=float(row["entry_price"]),
                stop_loss=float(row["stop_loss"]),
                take_profit=float(row["take_profit"]),
                opened_at=datetime.fromisoformat(row["opened_at"]),
            )
            for row in payload.get("positions", [])
        )
        return ShadowPortfolioState(
            base_equity_usd=float(payload["base_equity_usd"]),
            realized_pnl_usd=float(payload.get("realized_pnl_usd", 0.0)),
            equity_peak_usd=float(payload["equity_peak_usd"]),
            positions=positions,
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            schema_version=1,
        )

    def save(self, state: ShadowPortfolioState) -> None:
        payload = {
            "schema_version": state.schema_version,
            "base_equity_usd": state.base_equity_usd,
            "realized_pnl_usd": state.realized_pnl_usd,
            "equity_peak_usd": state.equity_peak_usd,
            "updated_at": state.updated_at.astimezone(timezone.utc).isoformat(),
            "positions": [
                {
                    "decision_id": position.decision_id,
                    "symbol": position.symbol,
                    "side": position.side.value,
                    "notional_usd": position.notional_usd,
                    "estimated_risk_usd": position.estimated_risk_usd,
                    "quantity": position.quantity,
                    "entry_price": position.entry_price,
                    "stop_loss": position.stop_loss,
                    "take_profit": position.take_profit,
                    "opened_at": position.opened_at.astimezone(timezone.utc).isoformat(),
                }
                for position in state.positions
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)
