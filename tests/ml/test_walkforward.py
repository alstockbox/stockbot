import numpy as np
import pandas as pd

from stockbot.ml.walkforward import walk_forward_predictions


def test_walk_forward_predictions_are_oos_only_and_repeatable():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=180, freq="B")
    X = pd.DataFrame(rng.normal(size=(180, 3)), index=idx, columns=["a", "b", "c"])
    y = pd.Series(0.02 * X["a"] - 0.01 * X["b"] + rng.normal(0, 0.004, 180), index=idx)
    a = walk_forward_predictions(X, y, train_size=80, test_size=20, embargo=2, model_type="ridge", seed=9)
    b = walk_forward_predictions(X, y, train_size=80, test_size=20, embargo=2, model_type="ridge", seed=9)
    assert a.iloc[:82].isna().all()
    assert a.notna().sum() > 0
    pd.testing.assert_series_equal(a, b)
