import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

random_state = 666
np.random.seed(random_state)


def load_data():
    # load data
    data_frame = pd.read_csv('./dataset.csv')
    print(data_frame.columns.values.tolist())  # ['V1', 'V2', 'labels']
    data = data_frame.values
    X = data[:, :2]
    y = data[:, 2]
    # shuffle
    shuffle_indexes = np.random.permutation(len(X))
    X, y = X[shuffle_indexes], y[shuffle_indexes]
    return X, y


if __name__ == '__main__':
    X, y = load_data()
    print(X.shape, y.shape)
    print(Counter(y))

    y = (y + 2) % 3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    kmeans = KMeans(n_clusters=3, random_state=random_state)
    kmeans.fit(X_train, y_train)

    y_predict = kmeans.predict(X_test)
    score = accuracy_score(y_true=y_test, y_pred=y_predict)
    print(score)  # 0.9966666666666667

    center = kmeans.cluster_centers_
    print(center)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='r')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='g')
    plt.scatter(X[y == 2, 0], X[y == 2, 1], color='b')
    plt.scatter(center[:, 0], center[:, 1], color='#CCEECC')
    plt.show()

    pass
