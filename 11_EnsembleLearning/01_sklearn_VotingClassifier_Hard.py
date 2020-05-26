import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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

    # LR
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_score = lr.score(X_test, y_test)
    print('LR score: {}'.format(lr_score))

    # SVC
    svc = SVC()
    svc.fit(X_train, y_train)
    svc_score = svc.score(X_test, y_test)
    print('SVC score: {}'.format(svc_score))

    # DT
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_score = dt.score(X_test, y_test)
    print('DT score: {}'.format(dt_score))

    # Ensemble Learning
    lr_predict = lr.predict(X_test)
    svc_predict = svc.predict(X_test)
    dt_predict = dt.predict(X_test)

    y_predict = np.array((lr_predict + svc_predict + dt_predict) > 1.5, dtype=int)
    hard_voting_score = accuracy_score(y_true=y_test, y_pred=y_predict)
    print('my hard voting score: {}'.format(hard_voting_score))

    vote_clf = VotingClassifier(
        voting='hard',
        estimators=[
            ('lr', LogisticRegression()),
            ('svc', SVC()),
            ('dt', DecisionTreeClassifier())
        ]
    )
    vote_clf.fit(X_train, y_train)
    score = vote_clf.score(X_test, y_test)
    print('sklearn hard voting score: {}'.format(score))

    pass
