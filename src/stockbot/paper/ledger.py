from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class PaperObservation:
    strategy_id: str
    timestamp: str
    net_return: float
    benchmark_return: float = 0.0
    turnover: float = 0.0
    cost_rate: float = 0.0
    fill_rate: float = 1.0
    signal_count: int = 0
    regime: str | None = None
    notes: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id is required")
        try:
            datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be ISO-8601") from exc
        for name, value in (
            ("net_return", self.net_return),
            ("benchmark_return", self.benchmark_return),
            ("turnover", self.turnover),
            ("cost_rate", self.cost_rate),
            ("fill_rate", self.fill_rate),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.turnover < 0.0 or self.cost_rate < 0.0:
            raise ValueError("turnover and cost_rate must be non-negative")
        if not 0.0 <= self.fill_rate <= 1.0:
            raise ValueError("fill_rate must be in [0,1]")
        if self.signal_count < 0:
            raise ValueError("signal_count must be non-negative")


class PaperTradingLedger:
    """Append-only JSONL ledger for future shadow/paper observations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, observation: PaperObservation) -> None:
        existing = self.records(strategy_id=observation.strategy_id)
        key = (observation.strategy_id, observation.timestamp)
        if any((row.strategy_id, row.timestamp) == key for row in existing):
            raise ValueError("duplicate paper observation for strategy/timestamp")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(observation), sort_keys=True) + "\n")

    def records(self, *, strategy_id: str | None = None) -> list[PaperObservation]:
        if not self.path.exists():
            return []
        rows: list[PaperObservation] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            observation = PaperObservation(**payload)
            if strategy_id is None or observation.strategy_id == strategy_id:
                rows.append(observation)
        rows.sort(key=lambda row: row.timestamp)
        return rows

    def strategy_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row.strategy_id for row in self.records()}))


def make_paper_observation(
    *,
    strategy_id: str,
    net_return: float,
    benchmark_return: float = 0.0,
    turnover: float = 0.0,
    cost_rate: float = 0.0,
    fill_rate: float = 1.0,
    signal_count: int = 0,
    regime: str | None = None,
    timestamp: str | None = None,
    notes: str | None = None,
) -> PaperObservation:
    return PaperObservation(
        strategy_id=strategy_id,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        net_return=float(net_return),
        benchmark_return=float(benchmark_return),
        turnover=float(turnover),
        cost_rate=float(cost_rate),
        fill_rate=float(fill_rate),
        signal_count=int(signal_count),
        regime=regime,
        notes=notes,
    )
