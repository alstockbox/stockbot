import numpy as np
import pandas as pd

from stockbot.ml.dataset import make_forward_return_dataset
from stockbot.ml.validation import walk_forward_splits


def test_forward_return_dataset_drops_unknown_future_labels():
    idx = pd.date_range("2025-01-01", periods=10, freq="B")
    close = pd.Series(np.arange(100.0, 110.0), index=idx)
    features = pd.DataFrame({"feature": np.arange(10.0)}, index=idx)
    X, y = make_forward_return_dataset(features, close, horizon=2)
    assert len(X) == 8
    assert y.iloc[0] == close.iloc[2] / close.iloc[0] - 1
    assert X.index.equals(y.index)


def test_walk_forward_splits_are_ordered_and_embargoed():
    splits = walk_forward_splits(n_samples=40, train_size=15, test_size=5, embargo=2)
    assert splits
    for train, test in splits:
        assert train.max() < test.min()
        assert test.min() - train.max() - 1 >= 2
        assert len(set(train).intersection(test)) == 0
