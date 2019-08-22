import numpy as np
from math import log2


class LabelEncoder:
    def encode(self, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        return y

    def decode(self, y):
        return self.classes_[y]


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


class DecisionTree:
    def build(self, X, y, score=entropy):
        self.root = self.build_subtree(X, y, score)
        return self

    def build_subtree(self, X, y, score=entropy):
        if len(X) == 0:
            return Node()
        current_score = score(y)

        # Set up some variables to track the best criteria
        best_gain = 0.0
        best_criteria = None
        best_sets = None

        column_count = len(X[0])
        for col in range(0, column_count):
            column_values = np.unique(X[:, col])

            for value in column_values:
                X1, y1, X2, y2 = self.divideset(X, y, col, value)

                # Information gain
                p = float(len(X1)) / len(X)
                gain = current_score - p * score(y1) - (1 - p) * score(y2)
                if gain > best_gain and len(X1) > 0 and len(X2) > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (X1, y1, X2, y2)
        # Create the sub branches
        if best_gain > 0:
            true_branch = self.build_subtree(best_sets[0], best_sets[1])
            false_branch = self.build_subtree(best_sets[2], best_sets[3])
            return Node(column=best_criteria[0], value=best_criteria[1],
                        true_branch=true_branch, false_branch=false_branch)
        else:
            return LabelEncoder().decode(np.argmax(uniquecounts(y)))

    def divideset(self, X, y, column, value):
        if isinstance(value, int) or isinstance(value, float):
            mask = X[:, column] >= value
        else:
            mask = X[:, column] == value

        return X[mask], y[mask], X[~mask], y[~mask]
