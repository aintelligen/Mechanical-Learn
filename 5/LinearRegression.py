import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error


class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.interception = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0],\
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        assert self.interception is not None and self.coef_ is not None, \
            "must fit before predict"

        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return 1 - mean_squared_error(y_test, y_predict) / np.var(y_test)

    def __repr__(self):
        return "LinearRegression()"
