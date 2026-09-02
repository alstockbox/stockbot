from stockbot.arena.registry import ChampionRegistry, PromotionCriteria


def test_challenger_requires_oos_evidence_before_promotion():
    registry = ChampionRegistry(PromotionCriteria(min_oos_samples=100, min_robustness=0.6, score_margin=0.05))
    registry.nominate("candidate", score=2.0, metrics={"max_drawdown": 0.1}, robustness=0.9, oos_samples=30)
    assert not registry.promote_if_qualified("candidate")


def test_qualified_challenger_can_become_champion():
    registry = ChampionRegistry(PromotionCriteria(min_oos_samples=100, min_robustness=0.6, score_margin=0.05))
    registry.nominate("candidate", score=2.0, metrics={"max_drawdown": 0.1}, robustness=0.9, oos_samples=150)
    assert registry.promote_if_qualified("candidate")
    assert registry.champion.name == "candidate"
