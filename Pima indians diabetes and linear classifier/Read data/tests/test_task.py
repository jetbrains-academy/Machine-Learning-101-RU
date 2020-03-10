import unittest
from ..task import read_data


class TestCase(unittest.TestCase):
    def test_X(self):
        X, y = read_data("./task/pima-indians-diabetes.csv")
        self.assertEquals((768, 9), X.shape, "Wrong train data length")

    def test_y(self):
        X, y = read_data("./task/pima-indians-diabetes.csv")
        self.assertEquals(768, len(y), "Wrong train data length")

    def test_y_value(self):
        X, y = read_data("./task/pima-indians-diabetes.csv")
        self.assertTrue(((y == -1) | (y == 1)).all(), "y array should contain only -1 and 1 values")
