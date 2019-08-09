import numpy as np
from math import log

import pandas as pd


def uniquecounts(y):
    return [np.count_nonzero(y == 0), np.count_nonzero(y == 1), np.count_nonzero(y == 2)]


class LabelEncoder:
    def encode(self, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        return y

    def decode(self, y):
        return self.classes_[y]


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

    def predict(self, x):
        return self.classify_subtree(x, self.root)

    def classify_subtree(self, x, sub_tree):
        if not isinstance(sub_tree, Node):
            return sub_tree
        else:
            v = x[sub_tree.column]
            if isinstance(v, int) or isinstance(v, float):
                if v >= sub_tree.value:
                    branch = sub_tree.true_branch
                else:
                    branch = sub_tree.false_branch
            else:
                if v == sub_tree.value:
                    branch = sub_tree.true_branch
                else:
                    branch = sub_tree.false_branch
            return self.classify_subtree(x, branch)


def read_data(path):
    data = pd.read_csv(path)
    y = data[['type']]
    X = data.drop('type', 1)
    y = LabelEncoder().encode(y)
    return X.as_matrix(), y, X.columns.values


if __name__ == '__main__':
    path = "halloween.csv"
    X, y, columns = read_data(path)

    tree = DecisionTree()
    tree = tree.build(X, y)
    print(tree.predict(X[0]))
