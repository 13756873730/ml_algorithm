import numpy as np
from sklearn import datasets
from tools.model_selection import train_test_split
from tools.metrics import r2_score

random_state = 666
np.random.seed(random_state)


class LinearRegression:
    def __init__(self):
        self._theta = None
        self.intercept_ = None
        self.coefficients_ = None

    def fit(self, X_train, y_train):
        Xb = np.ones(shape=(X_train.shape[0], X_train.shape[1] + 1))
        Xb[:, 1:] = X_train
        Xb_T = np.transpose(Xb)
        self._theta = np.linalg.inv(Xb_T.dot(Xb)).dot(Xb_T).dot(y_train)
        self.intercept_ = self._theta[0]
        self.coefficients_ = self._theta[1:]

    def _predict_one(self, X_row):
        Xb = np.ones(shape=(X_row.shape[0] + 1,))
        Xb[1:] = X_row
        y_predict = Xb.dot(self._theta)
        return y_predict

    def predict(self, X_predict):
        y_predict = np.array([self._predict_one(X_row) for X_row in X_predict])
        return y_predict

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        r_square_score = r2_score(y_test, y_predict)
        return r_square_score


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

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    score = lin_reg.score(X_test, y_test)
    print(score)  # 0.6524291219701941

    pass
