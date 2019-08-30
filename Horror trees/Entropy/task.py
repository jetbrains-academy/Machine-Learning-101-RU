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
