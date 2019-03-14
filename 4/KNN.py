import numpy as np
from math import sqrt
from collections import Counter

def KNN_classify(k,X_train, y_train , x):

    assert 1 <= k <= X_train.shape[0],"K must be valid"
    assert X_train.shape[0] == y_train.shape[0],\
        "the size of X_train must equal to the size of y_traiin"


    distances = [sqrt(np.sum((x_train-x)**2)) for x_train in X_train]
    nearest = np.argsort(distances)
    
    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)
    return votes.most_common(1)[0][0]


from sklearn import datasets as ds
iris = ds.load_iris()

X_train = iris.data[:,:2]
y_train = iris.target
KNN_classify(6,X_train,y_train,[4.5,5.3])