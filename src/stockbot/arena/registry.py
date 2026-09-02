from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionCriteria:
    min_oos_samples: int = 126
    min_robustness: float = 0.60
    score_margin: float = 0.05
    max_drawdown: float = 0.30


@dataclass(frozen=True)
class Candidate:
    name: str
    score: float
    metrics: dict[str, float]
    robustness: float
    oos_samples: int


class ChampionRegistry:
    def __init__(self, criteria: PromotionCriteria | None = None) -> None:
        self.criteria = criteria or PromotionCriteria()
        self.candidates: dict[str, Candidate] = {}
        self.champion: Candidate | None = None

    def nominate(
        self,
        name: str,
        score: float,
        metrics: dict[str, float],
        robustness: float,
        oos_samples: int,
    ) -> Candidate:
        candidate = Candidate(name, float(score), dict(metrics), float(robustness), int(oos_samples))
        self.candidates[name] = candidate
        return candidate

    def promote_if_qualified(self, name: str) -> bool:
        candidate = self.candidates[name]
        if candidate.oos_samples < self.criteria.min_oos_samples:
            return False
        if candidate.robustness < self.criteria.min_robustness:
            return False
        if float(candidate.metrics.get("max_drawdown", 0.0)) > self.criteria.max_drawdown:
            return False
        if self.champion is not None and candidate.score < self.champion.score + self.criteria.score_margin:
            return False
        self.champion = candidate
        return True
