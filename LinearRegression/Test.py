import pandas as pd
import numpy as np
from LinearRegression.RidgeRegression import RidgeRegression
import matplotlib.pyplot as plt

train_data = pd.read_table('ntumlone-hw4-hw4_train.dat', sep=' ', header=None)
test_data = pd.read_table('ntumlone-hw4-hw4_test.dat', sep=' ', header=None)
train_X = train_data.values[:, :2]
train_y = train_data.values[:, -1]
test_X = test_data.values[:, :2]
test_y = test_data.values[:, -1]


def test(X, y, test_X, test_y, alpha=1):
    rc = RidgeRegression(alpha=alpha)
    rc.train(X, y)
    e_in_y = rc.predict(X)
    e_out_y = rc.predict(test_X)
    print("e_in:", rc.error(y, e_in_y))
    print("e_out:", rc.error(test_y, e_out_y))


def val(alpha, split_test_size):
    split_index = int(train_X.shape[0] * (1 - split_test_size))
    val_train_X = train_X[:split_index]
    val_test_X = train_X[split_index:]
    val_train_y = train_y[:split_index]
    val_test_y = train_y[split_index:]

    test(val_train_X, val_train_y, val_test_X, val_test_y, alpha)


# for i in range(-10, 3):
#     print("alpha = ", 10 ** i)
#     val(alpha=10 ** i, split_test_size=0.4)
#     print('\n')

test(train_X, train_y, test_X, test_y, alpha=1)