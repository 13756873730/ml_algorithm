import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

random_state = 666
np.random.seed(random_state)


def load_data():
    # load data
    X = np.linspace(start=-5, stop=5, num=200)
    y = 2 * np.power(X, 2) + 3 * X + 4 + np.random.normal(0, 10, size=200)
    X = X.reshape(-1, 1)
    # shuffle
    shuffle_indexes = np.random.permutation(len(X))
    X, y = X[shuffle_indexes], y[shuffle_indexes]
    return X, y


def create_pipeline(degree=2, alpha=0.5):
    return Pipeline([
        ('polynomial_features', PolynomialFeatures(degree=degree)),
        ('standard_scaler', StandardScaler()),
        ('ridge_regression', Ridge(alpha=alpha))
    ])


if __name__ == '__main__':
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    poly_pipeline = create_pipeline(degree=2, alpha=0.5)
    poly_pipeline.fit(X_train, y_train)

    score = poly_pipeline.score(X_test, y_test)
    print(score)  # 0.8119664816421104

    plt.scatter(X, y, color='b')
    X_predict = np.linspace(start=-5, stop=5, num=50000).reshape(-1, 1)
    y_predict = poly_pipeline.predict(X_predict)
    plt.plot(X_predict, y_predict, color='r')
    plt.axis([-6, 6, -50, 100])
    plt.show()

    pass
