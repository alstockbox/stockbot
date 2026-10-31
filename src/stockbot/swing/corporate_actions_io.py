"""Strict offline JSON parser for point-in-time corporate-action research.

Payload is an ARRAY of explicit action events, never downloaded automatically.
No data licensing or corporate-action completeness assertion is inferred.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

from stockbot.swing.corporate_actions import CorporateAction

ALLOWED = {
    "symbol", "ex_date", "kind", "announced_at", "numerator",
    "denominator", "dividend_per_share_sek", "pay_date",
}
REQUIRED = {"symbol", "ex_date", "kind", "announced_at"}


def parse_action_records(payload: Any) -> tuple[CorporateAction, ...]:
    if not isinstance(payload, list):
        raise ValueError("corporate action payload must be a JSON list")
    if len(payload) > 100_000:
        raise ValueError("corporate action payload too large")
    events = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("corporate action entries must be JSON objects")
        missing = REQUIRED.difference(entry)
        extra = set(entry).difference(ALLOWED)
        if missing or extra:
            raise ValueError(f"invalid action schema: missing={sorted(missing)}, unknown={sorted(extra)}")
        kwargs = entry.copy()
        try:
            kwargs["ex_date"] = date.fromisoformat(entry["ex_date"])
            kwargs["announced_at"] = datetime.fromisoformat(
                entry["announced_at"].replace("Z", "+00:00")
            )
            if kwargs.get("pay_date") is not None:
                kwargs["pay_date"] = date.fromisoformat(kwargs["pay_date"])
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("invalid corporate action date or datetime") from exc
        events.append(CorporateAction(**kwargs))
    return tuple(events)
