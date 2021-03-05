import unittest
import numpy as np
from numpy.testing import assert_array_equal

from clustering import k_means
from distances import euclidean_distance
import clustering


class TestCase(unittest.TestCase):
    def test_kmeans_sizes(self):
        X = np.array([[0, 0], [0, 1], [0, 1]])
        labels, centers = k_means(X, 2, euclidean_distance)
        self.assertEqual(2, len(centers))
        self.assertEqual(3, len(labels))

    def test_kmeans_results(self):
        X = np.array([[0, 0], [0.5, 0], [0.5, 1], [1, 1]])
        expected_labels = [0, 0, 1, 1]
        expected_centers = np.array([[0, 0], [0, 1]])

        clustering.init_clusters = lambda x, y: np.array([[0, 0], [1, 1]])

        classification, clusters = k_means(X, 2, euclidean_distance)
        assert_array_equal(classification, expected_labels)
        assert_array_equal(clusters, expected_centers)
