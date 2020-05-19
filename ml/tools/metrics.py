import numpy as np


def accuracy_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    score = np.sum(y_true == y_pred) / len(y_true)
    return score


def r2_score():
    pass
