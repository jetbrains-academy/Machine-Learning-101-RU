class Node:
    def __init__(self, predicate_index=-1, predicate_value=None, true_branch=None, false_branch=None):
        self.column = predicate_index
        self.value = predicate_value
        self.true_branch = true_branch
        self.false_branch = false_branch
