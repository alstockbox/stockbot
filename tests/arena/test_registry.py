from stockbot.arena.registry import ChampionRegistry, PromotionCriteria


def metrics(drawdown=0.1, negative_month_rate=0.2, monthly_observations=12):
    return {
        "max_drawdown": drawdown,
        "negative_month_rate": negative_month_rate,
        "monthly_observations": monthly_observations,
    }


def test_challenger_requires_oos_evidence_before_promotion():
    registry = ChampionRegistry(PromotionCriteria(min_oos_samples=100, min_robustness=0.6, score_margin=0.05))
    registry.nominate("candidate", score=2.0, metrics=metrics(), robustness=0.9, oos_samples=30)
    assert not registry.promote_if_qualified("candidate")


def test_qualified_challenger_can_become_champion():
    registry = ChampionRegistry(PromotionCriteria(min_oos_samples=100, min_robustness=0.6, score_margin=0.05))
    registry.nominate("candidate", score=2.0, metrics=metrics(), robustness=0.9, oos_samples=150)
    assert registry.promote_if_qualified("candidate")
    assert registry.champion.name == "candidate"


def test_missing_monthly_stability_evidence_blocks_promotion():
    registry = ChampionRegistry()
    registry.nominate(
        "candidate",
        score=3.0,
        metrics={"max_drawdown": 0.05},
        robustness=0.9,
        oos_samples=200,
    )
    assert not registry.promote_if_qualified("candidate")


def test_higher_score_cannot_replace_champion_with_worse_month_stability():
    registry = ChampionRegistry(PromotionCriteria(score_margin=0.01))
    registry.nominate("champion", 2.0, metrics(), 0.9, 200)
    assert registry.promote_if_qualified("champion")

    registry.nominate(
        "reckless",
        5.0,
        metrics(negative_month_rate=0.30),
        0.9,
        200,
    )
    assert not registry.promote_if_qualified("reckless")
    assert registry.champion.name == "champion"
