from __future__ import annotations

import numpy as np


def walk_forward_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    embargo: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if min(n_samples, train_size, test_size) <= 0 or embargo < 0:
        raise ValueError("invalid split sizes")
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    train_end = train_size
    while True:
        test_start = train_end + embargo
        test_end = test_start + test_size
        if test_end > n_samples:
            break
        train = np.arange(0, train_end, dtype=int)
        test = np.arange(test_start, test_end, dtype=int)
        splits.append((train, test))
        train_end = test_end
    return splits
