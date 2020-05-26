import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    ada_boosting_clf = GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=1e-4,
        random_state=random_state
    )
    ada_boosting_clf.fit(X_train, y_train)

    score = ada_boosting_clf.score(X_test, y_test)
    print('score: {}'.format(score))

    pass
