import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

random_state = 666
np.random.seed(random_state)


def load_data():
    # load data
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    # shuffle
    shuffle_indexes = np.random.permutation(len(X))
    X, y = X[shuffle_indexes], y[shuffle_indexes]
    return X, y


if __name__ == '__main__':
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    # print(X.shape, X_train.shape, y_train.shape)  # (506, 13) (404, 13) (404,)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    score = lin_reg.score(X_test, y_test)
    print(score)  # 0.7721494841608652

    pass
