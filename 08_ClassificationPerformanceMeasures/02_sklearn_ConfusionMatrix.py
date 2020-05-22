import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

random_state = 666
np.random.seed(random_state)


def load_data():
    # load data
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    # data preprocessing for binary classifier
    # classifier by digit is zero: not zero for 0; zero for 1.
    y[y > 0] = 1
    y = 1 - y
    # shuffle
    shuffle_indexes = np.random.permutation(len(X))
    X, y = X[shuffle_indexes], y[shuffle_indexes]
    return X, y


def create_pipeline(degree=2, penalty='l2', C=1.0, multi_class='auto', solver='lbfgs'):
    return Pipeline([
        ('polynomial_features', PolynomialFeatures(degree=degree)),
        ('standard_scaler', StandardScaler()),
        ('logistic_regression', LogisticRegression(penalty=penalty, C=C, multi_class=multi_class, solver=solver))
    ])


if __name__ == '__main__':
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    log_reg = create_pipeline(degree=2, penalty='l2', C=1.0, multi_class='auto', solver='lbfgs')
    log_reg.fit(X_train, y_train)

    score = log_reg.score(X_test, y_test)
    print(score)  # 0.9972222222222222

    y_predict = log_reg.predict(X_test)
    confusion_matrix = confusion_matrix(y_true=y_test, y_pred=y_predict)
    print(confusion_matrix)

    precision = precision_score(y_true=y_test, y_pred=y_predict)
    recall = recall_score(y_true=y_test, y_pred=y_predict)
    print('precision={}, score={}'.format(precision, recall))

    f1_score = f1_score(y_true=y_test, y_pred=y_predict)
    print('f1_score={}'.format(f1_score))

    # ROC_Curve
    y_score = log_reg.decision_function(X_test)
    fprs, tprs, thresholds = roc_curve(y_true=y_test, y_score=y_score)
    # print(fprs.shape, tprs.shape, thresholds.shape)
    plt.plot(fprs, tprs)
    plt.show()

    # ROC_AUC_Score
    roc_auc_score = roc_auc_score(y_true=y_test, y_score=y_score)
    print('roc_auc_score={}'.format(roc_auc_score))

    pass
