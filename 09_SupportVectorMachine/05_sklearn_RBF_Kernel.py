import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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


def create_pipeline(kernel='rbf', gamma=1.0):
    return Pipeline([
        ('standard_scaler', StandardScaler()),
        ('svc', SVC(kernel=kernel, gamma=gamma))
    ])


if __name__ == '__main__':
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    svc_pipeline = create_pipeline(kernel='rbf', gamma=0.1)
    svc_pipeline.fit(X_train, y_train)

    score = svc_pipeline.score(X_test, y_test)
    print(score)  # 0.9666666666666667

    pass
