import numpy as np


class PLA:
    def __init__(self, learning_rate=1, n_iter=100):
        self.alpha = learning_rate
        self.n_iter = n_iter
        self.w = []
        self.pocket = []

    def _sign(self, X, y):
        signed = (X.dot(self.w) * y > 0).astype(np.int)
        signed[signed == 0] = -1

        return signed

    def _preprocess(self, X, y, pocket=False):
        X = np.array(X)
        X = np.concatenate((X, np.ones(X.shape[0]).reshape(-1, 1)), axis=1)
        y = np.array(y)
        self.w = np.zeros(X.shape[1])
        if pocket:
            self.pocket = np.zeros(X.shape[1])

        return X, y

    def update(self, xn, yn):
        delta = xn * yn * self.alpha
        self.w += delta

    def is_pocket(self, X, y):
        signed = X.dot(self.w) * y <= 0
        error = signed.sum()

        pocket_signed = X.dot(self.pocket) * y <= 0
        pocket_error = pocket_signed.sum()

        return error < pocket_error

    def train(self, X, y, pocket=False):
        X, y = self._preprocess(X, y, pocket)
        result = self._sign(X, y)

        for i in range(self.n_iter):
            if np.any(result < 0):
                # SGD
                neg_index = result < 0
                index = np.random.randint(X[neg_index].shape[0])

                xn = X[neg_index][index]
                yn = y[neg_index][index]
                self.update(xn, yn)
                if pocket and self.is_pocket(X, y):
                    self.pocket = self.w.copy()

                result = self._sign(X, y)
            else:
                # print(self.w)
                return i
        if pocket:
            self.w = self.pocket.copy()
            # print((X.dot(self.pocket)*y < 0).sum())

    def predict(self, X):
        X = np.concatenate((X, np.ones(X.shape[0]).reshape(-1, 1)), axis=1)
        predict_y = X.dot(self.w)
        # predict_y = predict_y + 0
        predict_y[predict_y > 0] = 1
        predict_y[predict_y <= 0] = -1

        return predict_y
