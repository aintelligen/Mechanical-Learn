import numpy as np

##数据集 ：分成训练（80%）和测试（20%）数据集 ，通过测试数据判断模型好坏  train test split
def train_test_split(X, y, test_ratio = 0.2, seed=None):
  #参数判断
  assert X.shape[0] == y.shape[0],\
    "the size of S must ve equal to the size of y"
  assert 0.0 <= test_ratio <= 1.0,\
    "test_ration must be valid"

  #随机种子
  if seed:
    np.random.seed(seed)
  
  #train 80%  test 20%  split
  shuffle_indexes = np.random.permutation(len(X))
  test_ratio = 0.2
  test_size = int(len(X)*test_ratio)
  test_indexes = shuffle_indexes[:test_size]
  train_indexes = shuffle_indexes[test_size:]

  #Fancy Indexing
  X_train = X[train_indexes]
  y_train = y[train_indexes]

  X_test = X[test_indexes]
  y_test = y[test_indexes]

  return X_train, X_test, y_train, y_test
    