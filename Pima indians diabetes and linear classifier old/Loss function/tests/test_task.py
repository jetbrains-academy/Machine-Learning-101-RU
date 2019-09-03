import unittest

import numpy as np
from numpy.ma.testutils import assert_array_almost_equal

from ..task import log_loss, sigmoid_loss


class TestCase(unittest.TestCase):
    def test_log_loss(self):
        a = np.array([1, 2, 3, 4, 5])
        loss, derivative = log_loss(a)
        expected_loss = np.array([0.45194108, 0.18311841, 0.07009673, 0.02618481, 0.0096882])
        expected_derivative = np.array([-0.26894142, -0.11920292, -0.04742587, -0.01798621, -0.00669285])
        assert_array_almost_equal(expected_loss, loss, 2)
        assert_array_almost_equal(expected_derivative, derivative, 2)

    def test_sigmoid_loss(self):
        a = np.array([1, 2, 3, 4, 5])
        loss, derivative = sigmoid_loss(a)
        expected_loss = np.array([0.53788284, 0.23840584, 0.09485175, 0.03597242, 0.0133857])
        expected_derivative = np.array([-0.393224, -0.209987, -0.090353, -0.035325, -0.013296])
        assert_array_almost_equal(expected_loss, loss, 2)
        assert_array_almost_equal(expected_derivative, derivative, 2)
