import numpy as np
import unittest

from numpy.ma.testutils import assert_array_equal

from ..task import read_image


class TestCase(unittest.TestCase):
    def test_read_image(self):
        image = read_image("./Read Image/tests/star.png")
        expected_star = np.loadtxt("./Read Image/tests/star.txt")
        assert_array_equal(expected_star, image)
