import numpy as np
import pandas as pd

from stockbot.research.bootstrap_uncertainty import evaluate_block_bootstrap_uncertainty


def test_block_bootstrap_is_deterministic_and_positive_for_strong_stream():
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0015, 0.004, 180), dtype=float)

    first = evaluate_block_bootstrap_uncertainty(
        returns,
        block_size=10,
        samples=100,
        confidence_level=0.90,
        seed=42,
    )
    second = evaluate_block_bootstrap_uncertainty(
        returns,
        block_size=10,
        samples=100,
        confidence_level=0.90,
        seed=42,
    )

    assert first == second
    assert first.observations == 180
    assert 0.0 <= first.confidence_score <= 1.0
    assert first.probability_positive_cagr > 0.5
    assert first.cagr.lower <= first.cagr.median <= first.cagr.upper
    assert first.sharpe.lower <= first.sharpe.median <= first.sharpe.upper


def test_block_bootstrap_rejects_too_short_return_series():
    try:
        evaluate_block_bootstrap_uncertainty(pd.Series([0.01] * 10), samples=50)
    except ValueError as exc:
        assert "20 return observations" in str(exc)
    else:
        raise AssertionError("expected ValueError")
