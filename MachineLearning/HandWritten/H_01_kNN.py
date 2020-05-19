import numpy as np
from collections import Counter
from sklearn import datasets
from MachineLearning.HandWritten.model_selection import train_test_split
from MachineLearning.HandWritten.metrics import accuracy_score

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


class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, p=2):
        self.n_neighbors = n_neighbors
        self.p = p
        self.X = None
        self.y = None

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train

    def _predict_one(self, X_predict_one):
        # minkowski p
        distance = np.array([
            # [Σ(|X_a - X_b| ^ p)] ^ (1/p)
            np.power(np.sum(np.power(np.abs(X_row - X_predict_one), self.p)), 1. / self.p)
            for X_row in self.X
        ])

        # distance sort ASC
        sort_indexes = np.argsort(distance)
        y_sort = self.y[sort_indexes[:self.n_neighbors]]

        counter = Counter(y_sort)
        vote = counter.most_common()
        y_predict = vote[0][0]
        return y_predict

    def predict(self, X_predict):
        y_predict = np.array([self._predict_one(X_predict_one=X_element) for X_element in X_predict])
        return y_predict

    def score(self, X_test, y_test):
        y_predict = self.predict(X_predict=X_test)
        score = accuracy_score(y_true=y_test, y_pred=y_predict)
        return score


pass

if __name__ == '__main__':
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    knn_clf = KNeighborsClassifier(n_neighbors=5, p=2)
    knn_clf.fit(X_train=X_train, y_train=y_train)

    score = knn_clf.score(X_test=X_test, y_test=y_test)
    print(score)
    pass
