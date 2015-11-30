import numpy as np


class NaivePLA:
    def __init__(self, learning_rate=1, n_iter=100, random=False, pocket=False):
        self.n_iter = n_iter
        self.alpha = learning_rate
        self.random = random
        self.w = []
        self.pocket = pocket
        self.pocket_w = []

    def _add_b(self, X):
        return np.concatenate((X, np.ones(X.shape[0]).reshape(-1, 1)), axis=1)

    def _init_w(self, X):
        self.w = np.zeros(X.shape[1])

    def _sign(self, x_n, y_n):
        if x_n.dot(self.w) * y_n > 0:
            return True
        return False

    @staticmethod
    def _error(w, X, y):
        return (X.dot(w) * y <= 0).sum()

    def _set_pocket_w(self, x_n, y_n, X, y):
        pre_error = self._error(self.w, X, y)
        error = self._error(self._update(x_n, y_n), X, y)

        if pre_error > error:
            self.pocket_w = self._update(x_n, y_n)
            return True
        return False

    def _update(self, x_n, y_n):
        return self.w + x_n * y_n * self.alpha

    def train(self, X, y):
        X = self._add_b(X)
        self._init_w(X)
        index = np.arange(X.shape[0])

        if self.random:
            np.random.shuffle(index)

        count = 0
        is_done = False

        iter_times = self.n_iter
        while not is_done and iter_times > 0:
            is_done = True
            for x_n, y_n in zip(X[index], y[index]):
                if not self._sign(x_n, y_n):
                    # update w n_iter times
                    if self.pocket:
                        self._set_pocket_w(x_n, y_n, X, y)
                    self.w = self._update(x_n, y_n)
                    is_done = False
                    iter_times -= 1
                    count += 1

        if self.pocket:
            self.w = self.pocket_w.copy()
        return count

    def predict(self, X):
        X = self._add_b(X)
        predict_y = X.dot(self.w)
        predict_y[predict_y <= 0] = -1
        predict_y[predict_y > 0] = 1

        return predict_y