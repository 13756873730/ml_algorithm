import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

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

    n_neighbors_list = range(1, 10)
    weights_list = ['uniform', 'distance']
    p_list = range(1, 10)

    best_score, best_n_neighbors, best_weights, best_p = 0, None, None, None
    for n_neighbors in n_neighbors_list:
        for weights in weights_list:
            for p in p_list:
                knn_clf = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    p=p
                )
                cv_score_list = cross_val_score(knn_clf, X_train, y_train)
                cv_score = np.mean(cv_score_list)
                if cv_score > best_score:
                    best_score, best_n_neighbors, best_weights, best_p = cv_score, n_neighbors, weights, p
    print(best_score, best_n_neighbors, best_weights, best_p)  # 0.9833333333333334 6 uniform 4
    knn_clf = KNeighborsClassifier(
        n_neighbors=best_n_neighbors,
        weights=best_weights,
        p=best_p
    )
    knn_clf.fit(X_train, y_train)
    score = knn_clf.score(X_test, y_test)
    print(score)  # 0.9666666666666667

    pass
