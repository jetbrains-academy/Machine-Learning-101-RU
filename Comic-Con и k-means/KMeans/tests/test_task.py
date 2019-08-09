import unittest

import numpy as np

from ..task import k_means


class TestCase(unittest.TestCase):
    def test_kmeans(self):
        def euclidean_distance(A, B):
            return np.sqrt(np.sum(np.square(A - B), axis=1))

        X = np.array([[0, 0], [0, 1], [0, 1]])
        centers, labels = k_means(X, 2, euclidean_distance)
        self.assertEqual(2, len(centers))
        self.assertEqual(3, len(labels))
