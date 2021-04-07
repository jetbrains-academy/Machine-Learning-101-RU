import numpy as np
from divide import Predicate
from node import Node

class DecisionTree:
    def build(self, X, y):
        self.root = self.build_subtree(X, y)
        pass

    def build_subtree(self, X, y):
        # get the most informative predicate using the get_best_predicate method
        predicate = # TODO

        # if such predicate is found:
        if predicate:
            # split the sample using the divide method of the class Predicate
            X1, y1, X2, y2 = # TODO
            # build subtrees recursively:
            true_branch = # TODO
            false_branch = # TODO
            # return tree as an instance of Node
            return Node(column=predicate.column, value=predicate.value,
                        true_branch=true_branch, false_branch=false_branch)
        # if the predicate was not found return the most common class label
        else:
            unique_y = np.unique(y, return_counts=True)
            return # TODO
        pass

    def get_best_predicate(self, X, y):
        best_predicate = None
        best_gain = 0.0
        column_count = len(X[0])

        # iterate over columns to look for values with the best predicate
        for column in range(0, column_count):
            # get unique values in the current column
            column_values = np.unique(X[:, column])
            #iterate over unique values in the column to calculate information gain for each using class Predicate
            for value in column_values:
                predicate = # TODO
                gain = # TODO
                if gain > best_gain:
                    # assign new values
                    best_predicate = # TODO
                    best_gain = # TODO

        return best_predicate

    def predict(self, x):
        return self.classify_subtree(x, self.root)

    def classify_subtree(self, x, sub_tree):
        pass
        # return sub_tree if it is a class label (not a Node instance)
        # if # TODO
            # return sub_tree
        # else:
            # take a value of the object x that is in the sub_tree column
            # v = # TODO
            # check if it is a numeric value
            # if isinstance(v, int) or isinstance(v, float):
                # the value v fits the numeric condition at the given node, proceed
                # to the true_branch, if not - to the false_branch
                # if v >= # TODO
                    # branch = # TODO
                # else:
                    # branch = # TODO
            # if v is not a numeric value - compare it with the nominal condition
            # at the node ad then do the same as in the if clause above
            # else:
                # if v == # TODO
                    # branch = # TODO
                # else:
                    # branch = # TODO
            # do the same recursively for the new branch
            # return # TODO

    # Below we defined a __repr__ method to ensure readable printing of Decision Tree instances
    def __repr__(self):
        return f'Decision Tree: \n{self.root};\n'