#函数写法
# import numpy as np
# from math import sqrt
# from collections import Counter

# def KNN_classify(k,X_train, y_train , x):

#     assert 1 <= k <= X_train.shape[0],"K must be valid"
#     assert X_train.shape[0] == y_train.shape[0],\
#         "the size of X_train must equal to the size of y_traiin"


#     distances = [sqrt(np.sum((x_train-x)**2)) for x_train in X_train]
#     nearest = np.argsort(distances)
    
#     topK_y = [y_train[i] for i in nearest[:k]]
#     votes = Counter(topK_y)
#     return votes.most_common(1)[0][0]


# from sklearn import datasets as ds
# iris = ds.load_iris()

# X_train = iris.data[:,:2]
# y_train = iris.target
# KNN_classify(6,X_train,y_train,[4.5,5.3])




#类写法
import numpy as np
from math import sqrt
from collections import Counter

class KNNClassifier:
    def __init__(self, k):
        assert k >= 1,"K must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None
        
    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self
    def predict(self, X_predict):
        assert self._X_train is not None and self._y_train is not None,\
            "must fit before predict"
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)
    def _predict(self, x):
        distances = [sqrt(np.sum((x_train-x)**2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]
        
    def __repr__(self):
        return "KNN(k=%d)" % self.k

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return sum(y_test == y_predict) / len(y_test)