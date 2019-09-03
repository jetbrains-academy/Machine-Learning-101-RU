import unittest
from ..task import read_data


class TestCase(unittest.TestCase):
    def test_X(self):
        X, y = read_data("./Read data/pima-indians-diabetes.csv")
        print(X.shape)
        print(y.shape)
        self.assertEquals(1000, X)
