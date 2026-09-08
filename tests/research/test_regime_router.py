import pandas as pd

from stockbot.domain.models import MarketRegime
from stockbot.ml.models import ModelConfig
from stockbot.research.regime_router import build_regime_router
from stockbot.research.regime_specialists import RegimeSpecialistResult
from stockbot.research.stress import evaluate_stress_suite


def _specialist(regime: MarketRegime, returns: pd.Series, score: float) -> RegimeSpecialistResult:
    stress = evaluate_stress_suite(returns)
    return RegimeSpecialistResult(
        regime=regime,
        horizon=5,
        model_config=ModelConfig("ridge", {"alpha": 1.0}, seed=7),
        metrics={"cagr": 0.1},
        robustness=0.8,
        oos_coverage=0.7,
        stress_report=stress,
        score=score,
        net_returns=returns,
        turnover_series=pd.Series(0.0, index=returns.index),
    )


def test_regime_router_uses_only_current_regime_specialist_returns():
    index = pd.date_range("2025-01-01", periods=90, freq="B", tz="UTC")
    regimes = pd.Series(
        [MarketRegime.BULL_TREND.value] * 30
        + [MarketRegime.BEAR_STRESS.value] * 30
        + [MarketRegime.NEUTRAL_CHOP.value] * 30,
        index=index,
    )
    bull = pd.Series(0.001, index=index)
    bear = pd.Series(-0.002, index=index)
    neutral = pd.Series(0.0002, index=index)
    specialists = {
        MarketRegime.BULL_TREND.value: _specialist(MarketRegime.BULL_TREND, bull, 1.2),
        MarketRegime.BEAR_STRESS.value: _specialist(MarketRegime.BEAR_STRESS, bear, 0.8),
        MarketRegime.NEUTRAL_CHOP.value: _specialist(MarketRegime.NEUTRAL_CHOP, neutral, 1.0),
    }

    report = build_regime_router(specialists, regimes)

    assert (report.returns.iloc[:30] == 0.001).all()
    assert (report.returns.iloc[30:60] == -0.002).all()
    assert (report.returns.iloc[60:] == 0.0002).all()
    assert report.regime_observations[MarketRegime.BULL_TREND.value] == 30
    assert 0.0 <= report.score <= 1.0
