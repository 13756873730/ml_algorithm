import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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

    standard_scaler = StandardScaler()
    standard_scaler.fit(X_train)
    X_train_standard = standard_scaler.transform(X_train)
    X_test_standard = standard_scaler.transform(X_test)

    linear_svc = LinearSVC(penalty='l2', C=1.0, multi_class='ovr', max_iter=1e5)
    linear_svc.fit(X_train_standard, y_train)

    score = linear_svc.score(X_test_standard, y_test)
    print(score)  # 0.9666666666666667

    pass
