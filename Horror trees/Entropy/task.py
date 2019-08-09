import numpy as np
from math import log


def uniquecounts(y):
    return [np.count_nonzero(y == 0), np.count_nonzero(y == 1), np.count_nonzero(y == 2)]


def entropy(y):
    log2 = lambda x: log(x) / log(2)
    results = uniquecounts(y)
    ent = 0.0
    for i, r in enumerate(results):
        if r == 0:
            continue
        p = float(results[i]) / len(y)
        ent -= p * log2(p)
    return ent


class Node:
    def __init__(self, column=-1, value=None, true_branch=None, false_branch=None):
        self.column = column
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
