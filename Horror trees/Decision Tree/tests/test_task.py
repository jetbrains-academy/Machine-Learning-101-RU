import unittest

import numpy as np

from ..task import DecisionTree, Node


class TestCase(unittest.TestCase):
    def test_root(self):
        X = np.array([[1, 2, 3],
                      [2, 2, 2],
                      [1, 4, 5]])
        y = np.array([1, 2, 1])

        tree = DecisionTree().build(X, y)
        self.assertEqual(1, tree.root.value)
        self.assertEqual(0, tree.root.column)

    def test_nodes(self):
        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        y = np.array([1, 2, 3])

        tree = DecisionTree().build(X, y)
        self.assertEqual(3, count_nodes(tree.root))


def count_nodes(node):
    counter = 0
    if isinstance(node, Node):
        counter += count_nodes(node.false_branch)
        counter += count_nodes(node.true_branch)
    else:
        counter += 1
    return counter
