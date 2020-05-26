import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

random_state = 666
np.random.seed(random_state)


def load_data():
    # load data
    X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=random_state)
    # shuffle
    shuffle_indexes = np.random.permutation(len(X))
    X, y = X[shuffle_indexes], y[shuffle_indexes]
    return X, y


if __name__ == '__main__':
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

    bagging_clf = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(
            max_depth=3
        ),
        n_estimators=500,
        max_samples=100,
        bootstrap=True
    )
    bagging_clf.fit(X_train, y_train)

    score = bagging_clf.score(X_test, y_test)
    print('bagging classifier score: {}'.format(score))

    pass
