from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PromotionCriteria:
    min_oos_samples: int = 126
    min_robustness: float = 0.60
    score_margin: float = 0.05
    max_drawdown: float = 0.30
    max_negative_month_rate: float = 0.50
    min_monthly_observations: int = 6
    max_drawdown_deterioration: float = 0.02
    max_negative_month_deterioration: float = 0.02


@dataclass(frozen=True)
class Candidate:
    name: str
    score: float
    metrics: dict[str, float]
    robustness: float
    oos_samples: int


def _finite_metric(metrics: dict[str, float], key: str) -> float | None:
    if key not in metrics:
        return None
    value = float(metrics[key])
    return value if math.isfinite(value) else None


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
        drawdown = _finite_metric(candidate.metrics, "max_drawdown")
        negative_month_rate = _finite_metric(candidate.metrics, "negative_month_rate")
        monthly_observations = _finite_metric(candidate.metrics, "monthly_observations")

        if candidate.oos_samples < self.criteria.min_oos_samples:
            return False
        if not math.isfinite(candidate.score):
            return False
        if candidate.robustness < self.criteria.min_robustness:
            return False
        if drawdown is None or drawdown > self.criteria.max_drawdown:
            return False
        if negative_month_rate is None or negative_month_rate > self.criteria.max_negative_month_rate:
            return False
        if monthly_observations is None or monthly_observations < self.criteria.min_monthly_observations:
            return False

        if self.champion is not None:
            if candidate.score < self.champion.score + self.criteria.score_margin:
                return False
            champion_dd = _finite_metric(self.champion.metrics, "max_drawdown")
            champion_loss_rate = _finite_metric(self.champion.metrics, "negative_month_rate")
            if champion_dd is not None and drawdown > champion_dd + self.criteria.max_drawdown_deterioration:
                return False
            if (
                champion_loss_rate is not None
                and negative_month_rate
                > champion_loss_rate + self.criteria.max_negative_month_deterioration
            ):
                return False

        self.champion = candidate
        return True
