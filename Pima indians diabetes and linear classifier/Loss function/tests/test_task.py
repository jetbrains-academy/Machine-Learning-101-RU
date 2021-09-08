import unittest

import numpy as np
from numpy.ma.testutils import assert_array_almost_equal

from ..loss_functions import log_loss, sigmoid_loss


class TestCase(unittest.TestCase):
    def test_log_loss(self):
        y = np.array([[0.5, 0.5], [0.1, 0.9], [0.01, 0.99],
                      [0.9, 0.1], [0.75, 0.25], [0.001, 0.999]])
        y_loss = log_loss(y)
        y_predicted = np.array([[[1.40529603, 1.40529603],
                                 [1.07393737, 1.79060654],
                                 [1.00723151, 1.88410338],
                                 [1.79060654, 1.07393737],
                                 [1.64015816, 1.19157871],
                                 [1.00072153, 1.89358157]],
                                [[-0.37754067, -0.37754067],
                                 [-0.47502081, -0.2890505],
                                 [-0.49750002, -0.27091208],
                                 [-0.2890505, -0.47502081],
                                 [-0.3208213, -0.4378235],
                                 [-0.49975, -0.26913808]]])
        assert_array_almost_equal(y_loss, y_predicted)

    def test_sigmoid_loss(self):
        y = np.array([[0.5, 0.5], [0.1, 0.9], [0.01, 0.99],
                      [0.9, 0.1], [0.75, 0.25], [0.001, 0.999]])
        y_loss = sigmoid_loss(y)
        y_predicted = np.array([[[0.75508134, 0.75508134],
                                 [0.95004163, 0.57810099],
                                 [0.99500004, 0.54182416],
                                 [0.57810099, 0.95004163],
                                 [0.6416426, 0.875647],
                                 [0.9995, 0.53827616]],
                                [[-0.47000742, -0.47000742],
                                 [-0.49875208, -0.41100061],
                                 [-0.4999875, -0.39503745],
                                 [-0.41100061, -0.49875208],
                                 [-0.43578999, -0.49226817],
                                 [-0.49999988, -0.39340555]]])
        assert_array_almost_equal(y_loss, y_predicted)
