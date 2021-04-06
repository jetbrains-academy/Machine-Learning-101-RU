import numpy as np
from divide import Predicate
from node import Node

class DecisionTree:
    def build(self, X, y):
        # TODO
        pass

    def build_subtree(self, X, y):
        # TODO
        pass

    def get_best_predicate(self, X, y):
        # TODO

        return best_predicate

    def predict(self, x):
        return self.classify_subtree(x, self.root)

    def classify_subtree(self, x, sub_tree):
        # TODO
        pass

    def __repr__(self):
        return f'Decision Tree: \n{self.root};\n'