import numpy as np
from sklearn import datasets
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

    bagging_clf = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(
            max_depth=3
        ),
        n_estimators=500,
        max_samples=100,
        bootstrap=True,
        oob_score=True
    )
    bagging_clf.fit(X, y)

    score = bagging_clf.oob_score_
    print('oob score: {}'.format(score))

    pass
