import numpy as np
import unittest

from numpy.ma.testutils import assert_array_equal

from task import read_image


class TestCase(unittest.TestCase):
    def test_read_image(self):
        image = read_image("./tests/star.png")
        expected_star = np.loadtxt("./tests/star.txt")
        assert_array_equal(expected_star, image)

    def test_shape(self):
        image = read_image("./task/superman-batman.png")
        self.assertEqual((786432, 3), image.shape)
