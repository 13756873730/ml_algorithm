import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope

random_state = 666
np.random.seed(random_state)


def load_data():
    # load data
    data_frame = pd.read_csv('./data.csv')
    print(data_frame.columns.values.tolist())  # ['x1', 'x2']
    data = data_frame.values
    X = data[:, :2]
    return X


if __name__ == '__main__':
    X = load_data()
    print(X.shape)  # (307, 2)

    elliptic_envelope = EllipticEnvelope(
        contamination=0.05
    )
    elliptic_envelope.fit(X)

    y_predict = elliptic_envelope.predict(X)

    plt.scatter(X[y_predict == 1, 0], X[y_predict == 1, 1], color='g')
    plt.scatter(X[y_predict == -1, 0], X[y_predict == -1, 1], color='r')
    plt.show()
    pass
