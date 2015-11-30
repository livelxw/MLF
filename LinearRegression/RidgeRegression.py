import numpy as np


class RidgeRegression:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.w = None

    def sign(self, x):
        sign_x = x.copy()
        sign_x[sign_x <= 0] = -1
        sign_x[sign_x > 0] = 1
        return sign_x

    def solve(self, X, y):
        return np.linalg.pinv(X.T.dot(X) + self.alpha * np.identity(X.shape[1])).dot(X.T).dot(y)

    def error(self, y, predict_y):
        return sum(y != predict_y) / y.size

    def train(self, X, y):
        X = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))
        self.w = self.solve(X, y)

    def predict(self, X):
        X = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))
        return self.sign(X.dot(self.w))
