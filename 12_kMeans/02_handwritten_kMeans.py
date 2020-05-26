import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from collections import Counter
from tools.model_selection import train_test_split
from tools.metrics import accuracy_score

random_state = 2
random.seed(random_state)
np.random.seed(random_state)


class KMeans:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None

    # calculate two vectors distance, default minkowski p=2
    @staticmethod
    def _distance(vector1, vector2, p=2):
        return np.power(np.sum(np.power(np.abs(vector1 - vector2), p)), 1 / p)

    def fit(self, X):
        # distribute all samples to cluster
        def distribute(X):
            y = np.empty(shape=(X.shape[0],))
            for row_index in range(X.shape[0]):
                row = X[row_index]
                min_distance_index = 0
                min_distance = self._distance(row, self.cluster_centers_[min_distance_index])
                for index in range(self.n_clusters):
                    cur_center = self.cluster_centers_[index]
                    cur_distance = self._distance(row, cur_center)
                    if cur_distance < min_distance:
                        min_distance_index, min_distance = index, cur_distance
                y[row_index] = min_distance_index
            return y

        # random cluster centers
        self.cluster_centers_ = X[np.random.randint(low=0, high=X.shape[0], size=self.n_clusters)]

        # empty label list
        y = distribute(X)

        # iterate until cluster centers steady
        while True:
            for index in range(self.n_clusters):
                mean = np.mean(X[y == index, :], axis=0)
                self.cluster_centers_[index] = mean
            old_y = y
            y = distribute(X)
            if (old_y == y).all():
                return self

    def _predict_one(self, row):
        min_distance_index = 0
        min_distance = self._distance(row, self.cluster_centers_[min_distance_index])
        for index in range(self.n_clusters):
            cur_center = self.cluster_centers_[index]
            cur_distance = self._distance(row, cur_center)
            if cur_distance < min_distance:
                min_distance_index, min_distance = index, cur_distance
        return min_distance_index

    def predict(self, X):
        return np.array([self._predict_one(row) for row in X])


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
    # print(X.shape, y.shape)  # (150, 4) (150,)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_train)

    y_predict = kmeans.predict(X_test)
    # 0不变，交换1和2
    y_predict[y_predict == 0] = 0
    y_predict[y_predict == 1] = 3
    y_predict[y_predict == 2] = 1
    y_predict[y_predict == 3] = 2

    score = accuracy_score(y_true=y_test, y_pred=y_predict)
    print(score)  # 0.8666666666666667

    center = kmeans.cluster_centers_
    print(center)

    pass
