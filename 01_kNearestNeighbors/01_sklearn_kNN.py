import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

random_state = 666
np.random.seed(random_state)


def load_data():
    # load data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # print(y)
    # shuffle
    shuffle_indexes = np.random.permutation(len(X))
    X, y = X[shuffle_indexes], y[shuffle_indexes]
    # print(y)
    return X, y


if __name__ == '__main__':
    X, y = load_data()
    # print(type(X), type(y))  # <class 'numpy.ndarray'>
    # print(X.shape, y.shape)  # (150, 4) (150,)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    # print(X_train.shape)  # (120, 4)

    knn_clf = KNeighborsClassifier(
        n_neighbors=5,
        metric='minkowski', p=2,
        weights='uniform'
    )
    knn_clf.fit(X=X_train, y=y_train)
    score = knn_clf.score(X=X_test, y=y_test)
    print(score)  # 0.9333333333333333
    pass
