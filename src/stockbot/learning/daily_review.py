from __future__ import annotations

from stockbot.llm.schemas import ResearchHypothesis


def fallback_hypotheses(context: dict) -> list[ResearchHypothesis]:
    metrics = context.get("metrics", {}) or {}
    regret = context.get("regret", {}) or {}
    ideas: list[ResearchHypothesis] = []

    cost_ratio = float(metrics.get("cost_ratio", 0.0) or 0.0)
    if cost_ratio >= 0.10:
        ideas.append(
            ResearchHypothesis(
                hypothesis="Trading costs may be consuming a material share of gross edge.",
                evidence=[f"Observed cumulative cost ratio is {cost_ratio:.4f}."],
                proposed_change="Test a higher entry hurdle and lower-turnover signal smoothing.",
                validation_plan=[
                    "Re-run walk-forward validation with identical signals and progressively stronger turnover penalties.",
                    "Compare net alpha, turnover and drawdown by market regime.",
                ],
                invalidation_criteria=[
                    "Reject if lower turnover reduces out-of-sample net alpha or materially worsens drawdown."
                ],
                category="execution_costs",
            )
        )

    negative_month_rate = float(metrics.get("negative_month_rate", 0.0) or 0.0)
    if negative_month_rate >= 0.30:
        ideas.append(
            ResearchHypothesis(
                hypothesis="The strategy is producing negative calendar months too frequently.",
                evidence=[f"Observed negative-month rate is {negative_month_rate:.2%}."],
                proposed_change="Test regime-conditioned exposure and stricter opportunity gating.",
                validation_plan=[
                    "Measure calendar-month loss frequency on purged out-of-sample folds.",
                    "Require the challenger to preserve or improve drawdown while reducing negative-month frequency.",
                ],
                invalidation_criteria=[
                    "Reject if monthly stability improves only in-sample or materially reduces risk-adjusted net return."
                ],
                category="monthly_stability",
            )
        )

    max_dd = float(metrics.get("max_drawdown", 0.0) or 0.0)
    if max_dd >= 0.08:
        ideas.append(
            ResearchHypothesis(
                hypothesis="Drawdown clustering may indicate weak regime adaptation.",
                evidence=[f"Maximum drawdown is {max_dd:.2%}."],
                proposed_change="Test lower strategy budgets during stress and high-volatility regimes.",
                validation_plan=[
                    "Evaluate the change across purged walk-forward folds.",
                    "Report return retained per unit of drawdown reduction.",
                ],
                invalidation_criteria=[
                    "Reject if drawdown improvement is not stable across folds or destroys risk-adjusted net return."
                ],
                category="regime_risk",
            )
        )

    regret_observations = int(regret.get("observations", 0) or 0)
    avoidable_loss = float(regret.get("avoidable_loss", 0.0) or 0.0)
    missed_gain = float(regret.get("missed_gain", 0.0) or 0.0)
    if regret_observations >= 3 and (avoidable_loss > 0.0 or missed_gain > 0.0):
        ideas.append(
            ResearchHypothesis(
                hypothesis="A causal alternative policy may have reduced realized decision regret.",
                evidence=[
                    f"Observed causal avoidable loss={avoidable_loss:.4f}, missed gain={missed_gain:.4f}."
                ],
                proposed_change="Test the pre-specified alternative policy as a challenger, without hindsight-only features.",
                validation_plan=[
                    "Replay the alternative using only information available at each historical decision time.",
                    "Validate with purged walk-forward splits before any shadow-stage promotion.",
                ],
                invalidation_criteria=[
                    "Reject if the improvement disappears after costs, embargo, or out-of-sample evaluation."
                ],
                category="decision_regret",
            )
        )

    if bool(context.get("regime_degradation", False)):
        ideas.append(
            ResearchHypothesis(
                hypothesis="Recent strategy expectancy appears weaker in the active regime.",
                evidence=["The monitoring context flags regime-specific degradation."],
                proposed_change="Reduce the degraded strategy weight and test alternative regime-conditioned allocations.",
                validation_plan=[
                    "Measure strategy expectancy by regime using out-of-sample observations only.",
                    "Shadow-test the challenger weighting before promotion.",
                ],
                invalidation_criteria=["Reject if the new weighting does not improve OOS research score."],
                category="ensemble",
            )
        )

    if not ideas:
        ideas.append(
            ResearchHypothesis(
                hypothesis="Feature interactions may contain incremental edge beyond the transparent baseline.",
                evidence=["No dominant cost, drawdown, regret or regime failure was flagged in the current review."],
                proposed_change="Train a deterministic ML challenger on causal features and future excess-return targets.",
                validation_plan=[
                    "Use ordered walk-forward splits with embargo.",
                    "Compare net OOS score against the current transparent baseline.",
                ],
                invalidation_criteria=[
                    "Reject if the challenger fails to beat the champion after costs and robustness penalties."
                ],
                category="machine_learning",
            )
        )
    return ideas
