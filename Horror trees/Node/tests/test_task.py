import unittest
from ..task import Node


class TestCase(unittest.TestCase):
    def test_true_branch(self):
        node = Node(1, 2, [1,2], [3,4])
        self.assertTrue(hasattr(node, "true_branch"), "Store true_branch in the true_branch field")

    def test_false_branch(self):
        node = Node(1, 2, [1,2], [3,4])
        self.assertTrue(hasattr(node, "false_branch"), "Store false_branch in the true_branch field")

    def test_fields(self):
        node = Node(1, 2, [1,2], [3,4])
        self.assertEqual(4, len(node.__dict__), "You should store all values passed to the Node object as field")
