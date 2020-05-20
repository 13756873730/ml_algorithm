import numpy as np
from sklearn import datasets
from tools.preprocessing import standardization
from tools.model_selection import train_test_split
from tools.metrics import r2_score

random_state = 666
np.random.seed(random_state)


class LinearRegression:
    def __init__(self):
        self._theta = None
        self._intercept = None
        self._coefficients = None

    def fit(self, X_train, y_train, n_iters=1e4):

        def loss(Xb, theta, y):
            # loss = 1/2 * Σ[(y - Xb*theta)^2]
            return 1 / 2 * np.sum(np.power(y - Xb.dot(theta), 2))

        def grad(Xb_i, theta, y_i):
            # grad = Xb.T * (Xb * theta - y)
            return Xb_i.T.dot(Xb_i.dot(theta) - y_i)

        def stochastic_gradient_descent(Xb, y, initial_theta, n_iters=n_iters):
            theta = initial_theta
            i_iter = 0
            while i_iter < n_iters:
                if i_iter % 100 == 0:
                    print(i_iter, loss(Xb, theta, y))
                random_row_index = np.random.randint(low=0, high=Xb.shape[0])
                grad_theta = grad(Xb[random_row_index], theta, y[random_row_index])
                t0, t1 = 5, 50
                learning_rate = t0 / (i_iter + t1)
                theta = theta - learning_rate * grad_theta
                i_iter += 1
            return theta

        # normalization
        X_train = standardization(X_train)
        # Xb
        Xb = np.hstack([np.ones(shape=(X_train.shape[0], 1)), X_train])
        # theta init
        initial_theta = np.zeros(shape=(Xb.shape[1],))
        # gradient descent
        theta = stochastic_gradient_descent(Xb, y_train, initial_theta)
        self._theta = theta
        self._intercept = self._theta[0]
        self._coefficients = self._theta[1:]

    def _predict_one(self, X_row):
        Xb = np.ones(shape=(X_row.shape[0] + 1,))
        Xb[1:] = X_row
        y_predict = Xb.dot(self._theta)
        return y_predict

    def predict(self, X_predict):
        # normalization
        X_predict = standardization(X_predict)
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
    lin_reg.fit(X_train, y_train, n_iters=1e5)

    score = lin_reg.score(X_test, y_test)
    print(score)  # 0.6521187479926799

    pass
