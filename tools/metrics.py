import numpy as np


def accuracy_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    score = np.sum(y_true == y_pred) / len(y_true)
    return score


def mean_squared_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    # mse = 1/m * Σ[(y_true - y_pred)^2]
    mse = 1 / len(y_true) * np.sum(np.power(y_true - y_pred, 2))
    return mse


def root_mean_squared_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    # rmse = 1/m * Σ[(y_true - y_pred)^2] ^ 1/2
    rmse = 1 / len(y_true) * np.power(np.sum(np.pow(y_true - y_pred, 2)), 1 / 2)
    return rmse


def mean_absolute_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    # mae = 1/m * Σ|y_true - y_pred|
    mae = 1 / len(y_true) * np.sum(np.abs(y_true - y_pred))
    return mae


def r2_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    ss_residual = mean_squared_error(y_true, y_pred)
    ss_total = np.var(y_true)
    r_square_score = 1 - ss_residual / ss_total
    return r_square_score


def _tn(y_true, y_pred):
    return np.sum((y_true == 0) & (y_pred == 0))


def _fp(y_true, y_pred):
    return np.sum((y_true == 0) & (y_pred == 1))


def _fn(y_true, y_pred):
    return np.sum((y_true == 1) & (y_pred == 0))


def _tp(y_true, y_pred):
    return np.sum((y_true == 1) & (y_pred == 1))


def confusion_matrix(y_true, y_pred):
    return np.array([
        [_tn(y_true, y_pred), _fp(y_true, y_pred)],
        [_fn(y_true, y_pred), _tp(y_true, y_pred)]
    ])


def precision_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fp = cm[0, 1]
    tp = cm[1, 1]
    if tp == 0:
        return 0
    return tp / (fp + tp)


def recall_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fn = cm[1, 0]
    tp = cm[1, 1]
    if tp == 0:
        return 0
    return tp / (fn + tp)


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * precision * recall / (precision + recall)
