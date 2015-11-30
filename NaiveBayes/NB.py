import numpy as np


class NaiveBayes:
    def __init__(self, lmda=1):
        self.lmda = lmda
        self.labels_prob = {}
        self.features_label_prob = {}

    def prob(self, vector, prop):
        return (vector[vector == prop].size + self.lmda) / (vector.size + np.unique(vector).size * self.lmda)

    def fit(self, X, y):
        for y_k in np.unique(y):
            self.labels_prob[y_k] = self.prob(y, y_k)

        for y_k in self.labels_prob.keys():
            self.features_label_prob[y_k] = {}
            for col_id in range(X.shape[1]):
                self.features_label_prob[y_k][col_id] = {}
                col = X[:, col_id][y == y_k]
                xs = np.unique(col)
                for x in xs:
                    self.features_label_prob[y_k][col_id][x] = self.prob(col, x)

        # print(self.labels_prob, self.features_label_prob)

    def predict(self, X):
        predict_labels_prob = np.ndarray([2, len(self.labels_prob)])
        y_index = 0
        for y in self.labels_prob.keys():
            prob = self.labels_prob[y]
            for col_id in range(X.size):
                prob *= self.features_label_prob[y][col_id][X[col_id]]
            predict_labels_prob[0, y_index] = y
            predict_labels_prob[1, y_index] = prob
            y_index += 1

        max_index = np.argmax(predict_labels_prob[1])
        return predict_labels_prob[0][max_index]
