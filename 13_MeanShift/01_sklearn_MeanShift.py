import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import estimate_bandwidth, MeanShift
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    mean_shift = MeanShift(bandwidth=estimate_bandwidth(X))
    mean_shift.fit(X_train)

    y_predict = mean_shift.predict(X_test)
    # 1不变，0和2互换
    y_predict[y_predict == 1] = 1
    y_predict[y_predict == 0] = 3
    y_predict[y_predict == 2] = 0
    y_predict[y_predict == 3] = 2

    score = accuracy_score(y_true=y_test, y_pred=y_predict)
    print(score)  # 0.9966666666666667

    pass
