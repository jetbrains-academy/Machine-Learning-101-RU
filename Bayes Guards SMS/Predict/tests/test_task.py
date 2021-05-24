import unittest

import numpy as np

from ..task import NaiveBayes


class TestCase(unittest.TestCase):
    def test_score(self):
        nb = NaiveBayes()
        X = np.array(['A great game', 'The election was over', 'Very clean match', 'A clean but forgettable game',
                          'It was a close election'])
        y = np.array(['Sports', 'Not sports', 'Sports', 'Sports', 'Not sports'])
        nb.fit(X, y)
        self.assertTrue(1, nb.score(X, y))

    def test_game(self):
        nb = NaiveBayes()
        X = np.array(['A great game', 'The election was over', 'Very clean match', 'A clean but forgettable game',
                          'It was a close election'])
        y = np.array(['Sports', 'Not sports', 'Sports', 'Sports', 'Not sports'])
        nb.fit(X, y)
        self.assertTrue("Sports", nb.predict(np.array(["Game"])))

    def test_election(self):
        nb = NaiveBayes()
        X = np.array(['A great game', 'The election was over', 'Very clean match', 'A clean but forgettable game',
                          'It was a close election'])
        y = np.array(['Sports', 'Not sports', 'Sports', 'Sports', 'Not sports'])
        nb.fit(X, y)
        self.assertTrue("Sports", nb.predict(np.array(["election"])))
