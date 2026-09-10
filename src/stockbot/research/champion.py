from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChampionState:
    experiment_id: str
    horizon: int
    model_name: str
    model_params: dict[str, Any]
    seed: int
    dataset_fingerprint: str
    factory_score: float
    promotion_score: float
    holdout_score: float
    promoted_at: str
    schema_version: int = 1


class JsonChampionStore:
    """Persistent champion state, separate from append-only experiment history."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> ChampionState | None:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return ChampionState(**payload)

    def save(self, state: ChampionState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(asdict(state), sort_keys=True, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


def make_champion_state(
    *,
    experiment_id: str,
    horizon: int,
    model_name: str,
    model_params: dict[str, Any],
    seed: int,
    dataset_fingerprint: str,
    factory_score: float,
    promotion_score: float,
    holdout_score: float,
) -> ChampionState:
    return ChampionState(
        experiment_id=str(experiment_id),
        horizon=int(horizon),
        model_name=str(model_name),
        model_params=dict(model_params),
        seed=int(seed),
        dataset_fingerprint=str(dataset_fingerprint),
        factory_score=float(factory_score),
        promotion_score=float(promotion_score),
        holdout_score=float(holdout_score),
        promoted_at=datetime.now(timezone.utc).isoformat(),
    )
