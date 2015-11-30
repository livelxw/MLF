from PLA import PLA
from NaivePLA import NaivePLA
import pandas as pd


def Q15_PLA():
    data = pd.read_table('./train_15.txt', header=None, sep=r'\s')

    train_X = data.values[:, :4]
    train_y = data.values[:, -1]

    pla = PLA()

    times = 2000
    total = 0
    for x in range(1, times):
        total += pla.train(train_X, train_y)

    print(total / times)


def Q18_PLA():
    train_data = pd.read_table('./train_18.txt', header=None, sep=r'\s')
    test_data = pd.read_table('./test_18.txt', header=None, sep=r'\s')
    train_X = train_data.values[:, :4]
    train_y = train_data.values[:, -1]
    test_X = test_data.values[:, :4]
    test_y = test_data.values[:, -1]

    pla = PLA(n_iter=50)

    total_error = 0
    times = 2000
    for x in range(times):
        pla.train(train_X, train_y, pocket=True)
        predict_y = pla.predict(test_X)
        total_error += predict_y[predict_y != test_y].size / test_y.size
        # print(x)

    print(total_error / times)


def Q15_NaivePLA(random=False, times=1, n_iter=100):
    data = pd.read_table('./train_15.txt', header=None, sep=r'\s')

    train_X = data.values[:, :4]
    train_y = data.values[:, -1]

    count = 0

    pla = NaivePLA(n_iter=n_iter, random=random)
    for i in range(times):
        count += pla.train(train_X, train_y)
    print(count / times)


def Q18_NaivePLA(random=True, times=2000, n_iter=50, pocket=True):
    train_data = pd.read_table('./train_18.txt', header=None, sep=r'\s')
    test_data = pd.read_table('./test_18.txt', header=None, sep=r'\s')
    train_X = train_data.values[:, :4]
    train_y = train_data.values[:, -1]
    test_X = test_data.values[:, :4]
    test_y = test_data.values[:, -1]

    pla = NaivePLA(n_iter=n_iter, random=random, pocket=pocket)
    total_error = 0
    for i in range(times):
        pla.train(train_X, train_y)
        predict_y = pla.predict(test_X)
        total_error += predict_y[predict_y != test_y].size / test_y.size

    print(total_error / times)

Q18_NaivePLA(random=True, times=200, n_iter=50, pocket=True)
