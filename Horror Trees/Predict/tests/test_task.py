import unittest

import numpy as np

from tree import DecisionTree


class TestCase(unittest.TestCase):
    def test_has_method(self):
        self.assertTrue(hasattr(DecisionTree, "predict"), "Implement method `predict`")

    def test_root(self):
        X = np.array([[1, 2, 3],
                      [2, 2, 2],
                      [1, 4, 5]])
        y = np.array([1, 2, 1])

        tree = DecisionTree().build(X, y)
        self.assertEqual(1, tree.predict(X[0]))
