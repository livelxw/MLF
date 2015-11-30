from LR.LogisticRegression import LogisticRegression
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
import pandas as pd


def test():
    lr = LogisticRegression(100)
    sk_lr = LR()
    # w = np.array([0, 0])
    X = np.array([[2, 1], [3, 1], [4, 1], [1, 2], [1, 3], [1, 4]])
    y = np.array([1, 1, 1, -1, -1, -1])

    lr.fit(X, y)
    sk_lr.fit(X, y)

    # print(lr.w)
    print(lr.predict(X), lr.e_out(X, y))

    # x = np.array([[1, -1, 0], [100, -100, 10]])
    # print(lr.sigmoid(x))


def q18():
    train_data = pd.read_table('./ntumlone-hw3-hw3_train.dat', header=None, sep=' ')
    test_data = pd.read_table('./ntumlone-hw3-hw3_test.dat', header=None, sep=' ')
    train_X = train_data.values[:, 1:21]
    train_y = train_data.values[:, -1]
    test_X = train_data.values[:, 1:21]
    test_y = train_data.values[:, -1]

    lr = LogisticRegression(2000, alpha=0.01, method='sgd')
    lr.fit(train_X, train_y)
    print(lr.predict(test_X))
    print(lr.e_out(test_X, test_y))

q18()
