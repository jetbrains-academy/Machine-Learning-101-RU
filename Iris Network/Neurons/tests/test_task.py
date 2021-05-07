import unittest
import numpy as np

from activation import sigmoid


class TestCase(unittest.TestCase):
    def test_sigmoid(self):
        self.assertEqual(0.01, round(sigmoid(-5), 2))
        self.assertEqual(0.05, round(sigmoid(-3), 2))
        self.assertEqual(0.27, round(sigmoid(-1), 2))
        self.assertEqual(0.73, round(sigmoid(1), 2))
        self.assertEqual(0.95, round(sigmoid(3), 2))
