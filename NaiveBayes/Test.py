import numpy as np
from NaiveBayes.NB import NaiveBayes


def test():
    X = np.array([[1, 's'],
                  [1, 'm'],
                  [1, 'm'],
                  [1, 's'],
                  [1, 's'],
                  [2, 's'],
                  [2, 'm'],
                  [2, 'm'],
                  [2, 'l'],
                  [2, 'l'],
                  [3, 'l'],
                  [3, 'm'],
                  [3, 'm'],
                  [3, 'l'],
                  [3, 'l']])
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

    nb = NaiveBayes()
    nb.fit(X, y)
    print(nb.predict(np.array([2, 's'])))

test()