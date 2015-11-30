import numpy as np


class LogisticRegression:
    def __init__(self, n_iter=10, alpha=0.1, method='gd'):
        self.alpha = alpha
        self.n_iter = n_iter
        self.w = None
        self.method = method

    def sign(self, scores):
        return np.where(scores >= 0.5, 1, -1)

    @staticmethod
    def sigmoid(s):
        return 1 / (1 + np.exp(-s))

    def gradient(self, w, X, y):
        s = -X.dot(w) * y
        theta = self.sigmoid(s)
        grad = np.average(-theta.reshape(-1, 1) * y.reshape(-1, 1) * X, axis=0)

        return grad

    def update(self, w, grad):
        return w - self.alpha * grad

    def descent(self, w, X, y):
        for i in range(self.n_iter):
            grad = self.gradient(w, X, y)
            # if (np.abs(grad) < 0.1).all():
            #     print(grad)
            #     break
            w = self.update(w, grad)
        return w

    def sgd(self, w, X, y):
        for i in range(self.n_iter):
            index = np.random.randint(X.shape[0])
            x_n = X[index]
            y_n = y[index]
            grad = self.gradient(w, x_n, y_n)
            w = self.update(w, grad)
        return w

    def e_out(self, X, y):
        return (np.sum(self.predict(X) != y) * 1.0) / y.shape[0]

    def score(self, X, y):
        return 1 - self.e_out(X, y)

    def fit(self, X, y):
        X = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))
        self.w = np.zeros(X.shape[1])
        if self.method is 'sgd':
            print('sgd')
            self.w = self.sgd(self.w, X, y)
        else:
            self.w = self.descent(self.w, X, y)

    def predict(self, X):
        X = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))

        return self.sign(self.sigmoid(X.dot(self.w)))
