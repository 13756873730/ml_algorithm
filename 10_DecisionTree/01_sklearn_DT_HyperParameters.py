import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

random_state = 666
np.random.seed(random_state)


def load_data():
    # load data
    X, y = datasets.make_moons(n_samples=200, noise=0.25)
    # shuffle
    shuffle_indexes = np.random.permutation(len(X))
    X, y = X[shuffle_indexes], y[shuffle_indexes]
    return X, y


if __name__ == '__main__':
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    dt_clf = DecisionTreeClassifier(
        criterion='gini',  # 不确定度的评判标准
        max_depth=3,  # 树的最大深度，默认为None不进行限制
        min_samples_split=5,  # 至少有多少个节点才进行划分，默认为2
        min_samples_leaf=3,  # 每个叶子节点至少包含多少个样本点，默认为1
        max_leaf_nodes=10  # 最多包含多少个叶子结点，默认为None不进行限制
    )
    dt_clf.fit(X_train, y_train)

    score = dt_clf.score(X_test, y_test)
    print(score)  # 0.9333333333333333

    pass
