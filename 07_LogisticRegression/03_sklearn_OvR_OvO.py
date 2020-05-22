import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

random_state = 666
np.random.seed(random_state)


def load_data():
    # load data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # shuffle
    shuffle_indexes = np.random.permutation(len(X))
    X, y = X[shuffle_indexes], y[shuffle_indexes]
    return X, y


if __name__ == '__main__':
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    log_reg = LogisticRegression(penalty='l2', C=1.0)

    # One vs Rest
    # log_reg_ovr = LogisticRegression(penalty='l2', C=1.0, multi_class='ovr')
    log_reg_ovr = OneVsRestClassifier(estimator=log_reg)
    log_reg_ovr.fit(X_train, y_train)
    score = log_reg_ovr.score(X_test, y_test)
    print('OvR score: {}'.format(score))  # 0.8666666666666667

    # One vs One
    # log_reg_ovo = LogisticRegression(penalty='l2', C=1.0, multi_class='multinomial')
    log_reg_ovo = OneVsOneClassifier(estimator=log_reg)
    log_reg_ovo.fit(X_train, y_train)
    score = log_reg_ovo.score(X_test, y_test)
    print('OvO score: {}'.format(score))  # 0.9666666666666667

    pass
