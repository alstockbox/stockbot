import numpy as np

from stockbot.ml.challenger import MLChallenger


def test_ml_challenger_is_repeatable_and_finite():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 4))
    y = 0.03 * X[:, 0] - 0.02 * X[:, 1] + rng.normal(scale=0.005, size=120)
    a = MLChallenger(model_type="ridge", seed=7).fit(X[:100], y[:100]).predict_score(X[100:])
    b = MLChallenger(model_type="ridge", seed=7).fit(X[:100], y[:100]).predict_score(X[100:])
    assert np.all(np.isfinite(a))
    np.testing.assert_allclose(a, b)
