import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression

random_state = 666
np.random.seed(random_state)


def load_data():
    # load data
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    # data preprocessing for binary classifier
    # classifier by digit is zero: not zero for 0; zero for 1.
    y[y > 0] = 1
    y = 1 - y
    # shuffle
    shuffle_indexes = np.random.permutation(len(X))
    X, y = X[shuffle_indexes], y[shuffle_indexes]
    return X, y


def create_pipeline(degree=2, penalty='l2', C=1.0, multi_class='auto', solver='lbfgs'):
    return Pipeline([
        ('polynomial_features', PolynomialFeatures(degree=degree)),
        ('standard_scaler', StandardScaler()),
        ('logistic_regression', LogisticRegression(penalty=penalty, C=C, multi_class=multi_class, solver=solver))
    ])


def tn(y_true, y_predict):
    return np.sum((y_true == 0) & (y_predict == 0))


def fp(y_true, y_predict):
    return np.sum((y_true == 0) & (y_predict == 1))


def fn(y_true, y_predict):
    return np.sum((y_true == 1) & (y_predict == 0))


def tp(y_true, y_predict):
    return np.sum((y_true == 1) & (y_predict == 1))


def confusion_matrix(y_true, y_predict):
    return np.array([
        [tn(y_true, y_predict), fp(y_true, y_predict)],
        [fn(y_true, y_predict), tp(y_true, y_predict)]
    ])


def precision(confusion_matrix):
    fp = confusion_matrix[0, 1]
    tp = confusion_matrix[1, 1]
    if tp == 0:
        return 0
    return tp / (fp + tp)


def recall(confusion_matrix):
    fn = confusion_matrix[1, 0]
    tp = confusion_matrix[1, 1]
    if tp == 0:
        return 0
    return tp / (fn + tp)


if __name__ == '__main__':
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    log_reg = create_pipeline(degree=2, penalty='l2', C=1.0, multi_class='auto', solver='lbfgs')
    log_reg.fit(X_train, y_train)

    score = log_reg.score(X_test, y_test)
    print(score)  # 0.9972222222222222

    y_predict = log_reg.predict(X_test)
    confusion_matrix = confusion_matrix(y_true=y_test, y_predict=y_predict)
    print(confusion_matrix)

    precision = precision(confusion_matrix)
    recall = recall(confusion_matrix)
    print('precision={}, score={}'.format(precision, recall))
    pass
