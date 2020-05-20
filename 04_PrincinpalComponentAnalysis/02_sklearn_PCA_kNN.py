import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

random_state = 666
np.random.seed(random_state)


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

    pca = PCA(n_components=0.9, random_state=random_state)
    pca.fit(X_train)
    print(pca.explained_variance_ratio_)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # kNN
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train_pca, y_train)
    score = knn_clf.score(X_test_pca, y_test)
    print(score)  # 0.9944444444444445

    pass
