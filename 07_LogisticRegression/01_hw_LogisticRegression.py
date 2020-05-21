import numpy as np
from sklearn import datasets
from tools.model_selection import train_test_split
from tools.metrics import accuracy_score

random_state = 666
np.random.seed(random_state)


class LogisticRegression:
    def __init__(self):
        self._w = None
        self.intercept_ = None
        self.coefficients_ = None
        pass

    @staticmethod
    def _sigmoid(t):
        return 1 / (1 + np.exp(-t))

    def fit(self, X_train, y_train, learning_rate=1e-4, n_iters=1e4):
        def loss(Xb, w, y):
            # p_hat = sigmoid(Xb·w)
            p_hat = self._sigmoid(Xb.dot(w))
            # loss = 1/m * Σ{ y*[-log(p_hat)] + (1-y)[-log(1-p_hat)] }
            loss = 1 / Xb.shape[0] * np.sum(
                y * (-1 * np.log(p_hat)) + (1 - y) * (-1 * np.log(1 - p_hat))
            )
            return loss

        def grad(Xb, w, y):
            # grad = 1/m * Xb.T·[sigmoid(Xb·w) - y]
            grad = 1 / Xb.shape[0] * Xb.T.dot(self._sigmoid(Xb.dot(w)) - y)
            return grad

        def sgd(Xb, y, w_initial, learning_rate=learning_rate, n_iters=n_iters):
            w = w_initial
            cur_iter = 0
            while cur_iter < n_iters:
                loss_pre = loss(Xb, w, y)
                w = w - learning_rate * grad(Xb, w, y)
                loss_cur = loss(Xb, w, y)
                if np.abs(loss_pre - loss_cur) < 1e-8:
                    break
                cur_iter += 1
            return w

        Xb = np.hstack([np.ones(shape=(X_train.shape[0], 1)), X_train])
        w_initial = np.random.random(size=Xb.shape[1])
        self._w = sgd(Xb, y_train, w_initial)
        self.intercept_ = self._w[0]
        self.coefficients_ = self._w[1:]

    def _predict_one(self, X_row):
        p_hat = self._sigmoid(X_row.dot(self._w))
        y_predict = int(p_hat > 0.5)
        return y_predict

    def predict(self, X_predict):
        X_predict = np.hstack([np.ones(shape=(X_predict.shape[0], 1)), X_predict])
        return np.array([
            self._predict_one(X_row) for X_row in X_predict
        ])

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        score = accuracy_score(y_true=y_test, y_pred=y_pred)
        return score

    pass


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

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train, learning_rate=1e-2, n_iters=1e3)

    score = log_reg.score(X_test, y_test)
    print(score)  # 1.0

    pass
