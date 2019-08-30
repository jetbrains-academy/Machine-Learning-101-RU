import numpy as np


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -(p * np.log2(p)).sum()


class Node:
    def __init__(self, column=-1, value=None, true_branch=None, false_branch=None):
        self.column = column
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch


class Predicate:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def divide(self, X, y):
        if isinstance(self.value, int) or isinstance(self.value, float):
            mask = X[:, self.column] >= self.value
        else:
            mask = X[:, self.column] == self.value

        return X[mask], y[mask], X[~mask], y[~mask]

    def information_gain(self, X, y):
        X1, y1, X2, y2 = self.divide(X, y)
        p = float(len(X1)) / len(X)
        gain = entropy(y) - p * entropy(y1) - (1 - p) * entropy(y2)
        return gain
