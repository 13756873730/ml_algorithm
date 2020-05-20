import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

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

    params = [
        {
            'n_neighbors': range(1, 10),
            'weights': ['uniform', 'distance'],
            'p': range(1, 10)
        }
    ]

    knn_clf = KNeighborsClassifier()
    grid_search_cv = GridSearchCV(knn_clf, param_grid=params)
    grid_search_cv.fit(X_train, y_train)
    print(grid_search_cv.best_params_)  # {'n_neighbors': 6, 'p': 4, 'weights': 'uniform'}

    print(grid_search_cv.best_estimator_)
    knn_clf = grid_search_cv.best_estimator_
    knn_clf.fit(X_train, y_train)

    score = knn_clf.score(X_test, y_test)
    print(score)  # 0.9666666666666667

    pass
