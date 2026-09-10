from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockbot.arena.experiments import ExperimentConfig, ModelExperimentResult, _evaluate_panel_predictions
from stockbot.data.panel import build_panel
from stockbot.ml.models import ModelConfig
from stockbot.research.objective import risk_adjusted_objective
from stockbot.research.stress import StressReport, evaluate_stress_suite


@dataclass(frozen=True)
class PortfolioPolicy:
    top_fraction: float
    weighting: str = "equal"
    commission_bps: float = 1.0
    slippage_bps: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0,1]")
        if self.weighting not in {"equal", "conviction"}:
            raise ValueError("weighting must be 'equal' or 'conviction'")
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs cannot be negative")


@dataclass(frozen=True)
class PolicyResult:
    policy: PortfolioPolicy
    score: float
    metrics: dict[str, float]
    robustness: float
    stress_report: StressReport
    net_returns: pd.Series
    turnover_series: pd.Series


@dataclass(frozen=True)
class PolicyArenaReport:
    results: tuple[PolicyResult, ...]
    best: PolicyResult
    baseline: PolicyResult | None


def evaluate_policy_arena(
    bars: pd.DataFrame,
    model_result: ModelExperimentResult,
    *,
    top_fractions: tuple[float, ...] = (0.10, 0.20, 0.30, 0.50),
    weightings: tuple[str, ...] = ("equal", "conviction"),
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
) -> PolicyArenaReport:
    """Reuse one OOS prediction stream across multiple portfolio-construction policies.

    This creates several counterfactual portfolio decisions without refitting the ML
    model. It is therefore much cheaper than treating execution-policy choices as new
    model-training experiments. Results remain research diagnostics until the selected
    policy is independently validated on a blind holdout.
    """

    if model_result.predictions is None or len(model_result.predictions.dropna()) == 0:
        raise ValueError("model result must contain OOS predictions")
    if not top_fractions:
        raise ValueError("at least one top_fraction is required")
    if not weightings:
        raise ValueError("at least one weighting mode is required")

    panel = build_panel(bars)
    model_config = ModelConfig(
        model_result.artifact.model_name,
        params=dict(model_result.artifact.model_params),
        seed=int(model_result.artifact.seed),
    )

    results: list[PolicyResult] = []
    baseline: PolicyResult | None = None
    for top_fraction in top_fractions:
        for weighting in weightings:
            policy = PortfolioPolicy(
                top_fraction=float(top_fraction),
                weighting=weighting,
                commission_bps=float(commission_bps),
                slippage_bps=float(slippage_bps),
            )
            experiment_config = ExperimentConfig(
                model=model_config,
                top_fraction=policy.top_fraction,
                commission_bps=policy.commission_bps,
                slippage_bps=policy.slippage_bps,
                weighting=policy.weighting,
            )
            metrics, robustness, net_returns, turnover = _evaluate_panel_predictions(
                panel,
                model_result.predictions,
                experiment_config,
            )
            stress = evaluate_stress_suite(net_returns, turnover=turnover)
            score = risk_adjusted_objective(
                metrics,
                robustness=robustness,
                oos_coverage=model_result.oos_coverage,
                stress_score=stress.score,
            )
            result = PolicyResult(
                policy=policy,
                score=float(score),
                metrics={key: float(value) for key, value in metrics.items()},
                robustness=float(robustness),
                stress_report=stress,
                net_returns=net_returns,
                turnover_series=turnover,
            )
            results.append(result)
            if policy.top_fraction == 0.30 and policy.weighting == "equal":
                baseline = result

    results.sort(key=lambda item: item.score, reverse=True)
    if not results:
        raise ValueError("policy arena produced no results")
    return PolicyArenaReport(
        results=tuple(results),
        best=results[0],
        baseline=baseline,
    )
