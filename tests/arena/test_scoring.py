from stockbot.arena.scoring import research_score


def test_reward_rejects_fake_return_created_by_fragile_risk():
    safer = research_score(
        {"cagr": 0.30, "sharpe": 1.8, "calmar": 2.0, "max_drawdown": 0.15, "cvar_95": 0.02, "turnover": 1.0},
        robustness=0.85,
    )
    reckless = research_score(
        {"cagr": 0.65, "sharpe": 0.7, "calmar": 0.7, "max_drawdown": 0.65, "cvar_95": 0.15, "turnover": 12.0},
        robustness=0.25,
    )
    assert safer > reckless
