import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from tools.preprocessing import standardization
from tools.model_selection import train_test_split

random_state = 666
np.random.seed(random_state)


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None

    def fit(self, X_train, learning_rate=1e-4, n_iters=1e4):
        def demean(X):
            return X - np.mean(X, axis=0)

        def utility(X, w):
            # utility = 1/2m * Σ[(X*w)^2]
            return 1 / (2 * X.shape[0]) * np.sum(np.power(X.dot(w), 2))

        def grad(X, w):
            # grad = 1/m * X.T * (X * w)
            return 1 / X.shape[0] * X.T.dot(X.dot(w))

        def bga(X, initial_w, learning_rate=learning_rate, n_iters=n_iters):
            def direction(w):
                return w / np.linalg.norm(w)

            w = direction(initial_w)
            i_iter = 0
            while i_iter < n_iters:
                utility_pre = utility(X, w)
                w = direction(w + learning_rate * grad(X, w))
                utility_cur = utility(X, w)
                if utility_cur - utility_pre < 1e-8:
                    break
                i_iter += 1
            return w

        # normalization
        X_train = standardization(X_train)
        # demean
        X_demean = demean(X_train)
        self.components_ = np.empty(shape=(self.n_components, X_demean.shape[1]))
        for index in range(self.n_components):
            # w init
            initial_w = np.array(np.random.random(size=X_demean.shape[1]))
            # batch gradient ascent
            w = bga(X_demean, initial_w, learning_rate, n_iters)
            self.components_[index, :] = w
            X_demean = X_demean - X_demean.dot(w).reshape(-1, 1) * w
        return self

    def transform(self, X_mn):
        X_mk = X_mn.dot(self.components_.T)
        return X_mk

    def inverse_transform(self, X_mk):
        X_mn = X_mk.dot(self.components_)
        return X_mn


def load_data():
    # load data
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    # shuffle
    shuffle_indexes = np.random.permutation(len(X))
    X, y = X[shuffle_indexes], y[shuffle_indexes]
    return X, y


if __name__ == '__main__':
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    pca = PCA(n_components=20)
    pca.fit(X_train, learning_rate=1e-2, n_iters=1e4)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # kNN
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train_pca, y_train)
    score = knn_clf.score(X_test_pca, y_test)
    print(score)  # 0.9805555555555555

    pass
