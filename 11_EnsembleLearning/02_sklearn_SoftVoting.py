import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier


def load_data():
    # load data
    X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
    # shuffle
    shuffle_indexes = np.random.permutation(len(X))
    X, y = X[shuffle_indexes], y[shuffle_indexes]
    return X, y


if __name__ == '__main__':
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Ensemble Learning
    vote_clf = VotingClassifier(
        voting='hard',
        estimators=[
            ('lr', LogisticRegression()),
            ('svc', SVC(probability=True)),
            ('dt', DecisionTreeClassifier())
        ]
    )
    vote_clf.fit(X_train, y_train)
    score = vote_clf.score(X_test, y_test)
    print('sklearn soft voting score: {}'.format(score))

    pass
