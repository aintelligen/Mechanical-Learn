import numpy as np


class PCA:
    def __init__(self, n_components):
        assert n_components >= 1, "n_components must be valid"
        self.n_components = n_components
        self.components_ = None

    def fit(self, X, eta=0.01, n_iters=1e4):
        # 获得数据集X的前N个主成分
        assert self.n_components <= X.shape[1],\
            "n_components must not be greater that the feature number of X"

        def demean(X):
            return X-np.mean(X, axis=0)

        def f(w, X):
            return np.sum((X.dot(w)**2)) / len(X)

        def df_math(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def df_debug(w, X, epsilon=0.0001):
            res = np.empty(len(w))
            for i in range(len(w)):
                w_1 = w.copy()
                w_1[i] += epsilon
                w_2 = w.copy()
                w_2[i] -= epsilon
                res[i] = (f(w_1, X) - f(w_2, X)) / (2*epsilon)
            return res

        def direction(w):
            return w / np.linalg.norm(w)
        # 第一个成分

        def first_component(X_b, initial_w, eta, n_inters=1e4, epsilon=1e-8):
            w = direction(initial_w)
            i_inter = 0
            while i_inter < n_inters:
                gradient = df_math(w, X_b)
                last_w = w
                w = w + eta * gradient
                w = direction(w)  # 注意1： 每次求一个单位方向
                i_inter += 1
                if(abs(f(w, X_b) - f(last_w, X_b)) < epsilon):
                    break
            return w
        # 处理数据
        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))

        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca, initial_w, eta, n_iters)
            self.components_[i, :] = w
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

        return self

    def transform(self, X):
        # 将给定的X，映射到各个主成分分量中
        assert X.shape[1] == self.components_.shape[1]
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        # 将给定的X，反向映射回原来的特征空间
        assert X.shape[1] == self.components_.shape[0]
        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components
