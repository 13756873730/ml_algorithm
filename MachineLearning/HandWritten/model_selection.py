import numpy as np


def train_test_split(X, y, test_size=0.25, random_state=None):
    assert X.shape[0] == y.shape[0]
    # shuffle
    if random_state is not None:
        np.random.seed(random_state)
    shuffle_indexes = np.random.permutation(len(X))

    # split
    train_count = int(len(X) * (1 - test_size))
    train_indexes = shuffle_indexes[:train_count]
    test_indexes = shuffle_indexes[train_count:]

    return X[train_indexes], X[test_indexes], y[train_indexes], y[test_indexes]
