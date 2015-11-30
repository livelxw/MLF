import numpy as np
from KNN.KNearestNeighbors import KNearestNeighborsClassifier


def kdtree_test():
    X = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    y = np.array([1, 1, 1, 0, 0, 0])
    knn = KNearestNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    # print_node(knn.root)

    x = np.array([5, 4])
    nn = knn.predict(x)
    print(nn)


def print_node(node):
    if not node:
        return
    print(node.x)
    print_node(node.left)
    print_node(node.right)


kdtree_test()
