import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

random_state = 666
np.random.seed(random_state)


def load_data():
    # load data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # binary classifier
    X = X[y <= 1]
    y = y[y <= 1]
    # shuffle
    shuffle_indexes = np.random.permutation(len(X))
    X, y = X[shuffle_indexes], y[shuffle_indexes]
    return X, y


if __name__ == '__main__':
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    log_reg = LogisticRegression(penalty='l2')
    log_reg.fit(X_train, y_train)

    score = log_reg.score(X_test, y_test)
    print(score)  # 1.0

    pass
