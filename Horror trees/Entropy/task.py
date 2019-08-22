import numpy as np
from math import log2


def entropy(y):
    _, results = np.unique(y, return_counts=True)
    ent = 0.0
    for i, r in enumerate(results):
        p = float(results[i]) / len(y)
        ent -= p * log2(p)
    return ent


class Node:
    def __init__(self, column=-1, value=None, true_branch=None, false_branch=None):
        self.column = column
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
