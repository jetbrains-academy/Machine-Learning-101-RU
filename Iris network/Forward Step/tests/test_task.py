import unittest

from ..task import NeuralNetwork


class TestCase(unittest.TestCase):
    def test_add(self):
        nn = NeuralNetwork()
        self.assertEqual(sum(1, 2), 3, msg="adds 1 + 2 to equal 3")
