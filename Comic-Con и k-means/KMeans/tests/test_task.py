import unittest

import numpy as np

from ..task import k_means


class TestCase(unittest.TestCase):
    def test_kmeans(self):
        X = np.array([[0, 0], [0, 1], [0, 1]])
        centers, labels = k_means(X, n_clusters=2)
        self.assertEqual(2, len(centers))
        self.assertEqual(3, len(labels))
