from calculate_entropy import entropy

class Predicate:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def divide(self, X, y):
        # check if the value is numeric and create a boolean filter array
        # based on the "greater or equal to" condition
        if # TODO :
            mask = # TODO
        # if the value is not numeric (int or float), create the array based on the
        # "equal to" condition
        else:
            mask = # TODO
        # return the results in the following order: X1, y1, X2, y2
        return # TODO

    # this method is to be implemented in the task "Information Gain"
    def information_gain(self, X, y):
        pass
        # use the divide method to split the sample
        # X1, y1, X2, y2 = # TODO
        # calculate the fraction of X1 in the whole dataset
        # p = # TODO
        # use entropy function you wrote earlier and the formula
        # from the task to calculate information gain
        # gain = # TODO
        # return gain