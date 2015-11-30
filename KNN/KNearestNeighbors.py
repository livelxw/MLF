from operator import itemgetter
import numpy as np
from scipy.stats import mode


class KNearestNeighborsClassifier:
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.root = None
        self.k_dims = 0

    class Node:
        def __init__(self, x, y, depth):
            self.x = x
            self.y = y
            self.depth = depth
            self.left = None
            self.right = None

    class BPQ:
        def __init__(self, length=10, hold_max=False):
            self.data = []
            self.length = length
            self.hold_max = hold_max

        def append(self, x, distance, label):
            self.data.append((x.tolist(), distance, label))
            self.data.sort(key=itemgetter(1), reverse=self.hold_max)
            self.data = self.data[:self.length]

        def get_X(self):
            return [item[0] for item in self.data]

        def get_label(self):
            return [item[2] for item in self.data]

        def get_distance(self):
            return [item[1] for item in self.data]

        def get_threshold(self):
            return np.inf if len(self.data) == 0 else self.data[-1][1]

        def full(self):
            return len(self.data) >= self.length

    def _kd_tree_init(self, X, y):
        self.k_dims = X.shape[1]
        self.root = self._split(X, y, 0)

    def _split(self, x_s, y_s, depth):
        if x_s.size <= 0:
            return

        dim = depth % self.k_dims
        sorted_indexes = np.argsort(x_s[:, dim])
        split_point_index = sorted_indexes[int(sorted_indexes.size / 2)]

        split_point_node = self.Node(x=x_s[split_point_index], y=y_s[split_point_index], depth=depth)

        x_left = x_s[x_s[:, dim] < split_point_node.x[dim]]
        x_right = x_s[x_s[:, dim] > split_point_node.x[dim]]
        y_left = y_s[x_s[:, dim] < split_point_node.x[dim]]
        y_right = y_s[x_s[:, dim] < split_point_node.x[dim]]

        split_point_node.left = self._split(x_s=x_left, y_s=y_left, depth=depth + 1)
        split_point_node.right = self._split(x_s=x_right, y_s=y_right, depth=depth + 1)

        return split_point_node

    @staticmethod
    def _get_distance(a, b):
        return np.linalg.norm(a - b)

    def _search(self, x, node, queue):
        if node is not None:
            cur_distance = self._get_distance(x, node.x)
            if not queue.full() or cur_distance < queue.get_threshold():
                queue.append(node.x, cur_distance, node.y)
            axis = node.depth % self.k_dims
            search_left = False

            if x[axis] < node.x[axis]:
                search_left = True
                queue = self._search(x, node.left, queue)
            else:
                queue = self._search(x, node.right, queue)

            if np.abs(node.x[axis] - x[axis]) < queue.get_threshold():
                if search_left:
                    queue = self._search(x, node.right, queue)
                else:
                    queue = self._search(x, node.left, queue)

        return queue

    def get_nearest_neighbors(self, x):
        if self.root is not None:
            knn_queue = self.BPQ(length=self.n_neighbors)
            knn_queue = self._search(x, self.root, knn_queue)
            # nearest_point = self._search(x, self.root, self.root.x)
            return knn_queue.get_X(), knn_queue.get_label()

    def fit(self, X, y):
        self._kd_tree_init(X, y)

    def predict(self, x):
        X, y = self.get_nearest_neighbors(x)
        return mode(y).mode[0]
