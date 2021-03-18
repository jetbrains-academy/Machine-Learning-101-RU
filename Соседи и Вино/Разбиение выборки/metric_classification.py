import numpy as np


def knn(X_train, y_train, X_test, k, dist):
    # The function will return the class for x based on its neighbours from the X_train
    # sample
    def classify_single(x):
        # Here we create an array of distances from x to each of the X_train objects
        dists = #TODO
        # This array will contain indices of k nearest to the x objects. NumPy.argpartition
        # might be useful
        indices = #TODO
        # The function return the most frequent class among those in y_train represented
        # by the indices
        return #TODO

    return [classify_single(x) for x in X_test]