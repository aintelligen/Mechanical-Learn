import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error


class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0],\
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    # 批量梯度下降法  稳定

    def fit_gd(self,  X_train, y_train, eta=0.01, n_inters=1e4):
        assert X_train.shape[0] == y_train.shape[0],\
            "the size of X_train must be equal to the size of y_train"

        # 计算损失函数
        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta))**2) / len(y)
            except:
                return float('inf')

        # 计算梯度
        def dJ(theta, X_b, y):
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(X_b)

        # 梯度下降法
        def gradient_descent(X_b, y, initial_theta, eta, n_inters=1e3, epsilon=1e-8):
            theta = initial_theta
            i_inter = 0
            while i_inter < n_inters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                i_inter += 1
                if(abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(
            X_b, y_train, initial_theta, eta, n_inters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    # 随机梯度下降法  不稳定
    def fit_sgd(self,  X_train, y_train, n_inters=5, t0=5, t1=50):

        # 计算梯度
        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2.

        # 梯度下降法
        def sgd(X_b, y, initial_theta, n_inters=1e4, t0=5, t1=50):
            def learning_rate(t):
                return t0/(t+t1)

            theta = initial_theta
            i_inter = 0
            m = len(X_b)
            for cur_inter in range(n_inters):
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(m):
                    grandient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_inter*m + i) * grandient
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.random.randn(X_b.shape[1])
        self._theta = sgd(
            X_b, y_train, initial_theta, n_inters, t0, t1)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

    def predict(self, X_predict):
        assert self.intercept_ is not None and self.coef_ is not None, \
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
