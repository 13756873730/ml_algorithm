import numpy as np


def normalize(X):
    # X = (X - X_min) / (X_max - X_min)
    X = np.array(X, dtype=float)
    for col_index in range(X.shape[1]):
        X_col = X[:, col_index]
        X_min = np.min(X_col)
        X_max = np.max(X_col)
        X_col = (X_col - X_min) / max(1e-8, (X_max - X_min))
        X[:, col_index] = X_col
    return X


def standardization(X):
    # X = (X - X_mean) / X_std
    X = np.array(X, dtype=float)
    for col_index in range(X.shape[1]):
        X_col = X[:, col_index]
        X_mean = np.mean(X_col)
        X_std = np.std(X_col)
        X_col = (X_col - X_mean) / max(1e-8, X_std.item())
        X[:, col_index] = X_col
    return X
