import numpy as np


def precision_recall(y_pred, y_test):
    class_precision_recall = []
    # Here we calculate precision and recall for each of the unique classes of the
    # testing sample
    for c in np.unique(y_test):
        # Here we evaluate tp for the class
        tp = #TODO
        # Here we evaluate fp for the class
        fp = #TODO
        # Here we evaluate fn for the class
        fn = #TODO
        # Here we calculate precision for the class
        precision = #TODO
        # Here we calculate recall for the class
        recall =#TODO
        # Here we add a tuple containing a class, its precision and recall to the resulting array
        class_precision_recall.append((c, precision, recall))
    # The returned value would be an array of tuples containing unique classes and precision/ recall
    # values corresponding to them, like this: [(class 1, precision 1, recall 1), ...
    # ... , (class n, precision n, recall n)
    return class_precision_recall


def print_precision_recall(result):
    for c, precision, recall in result:
        print("class:", c, "\nprecision:", precision, "\nrecall:", recall, "\n")