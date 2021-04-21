# Here we defined the Node class to store tree nodes with the following attributes:
# - the predicate, by which the dataset is split in this node, stored as a number of
# a column (trait) and a value with which to compare;
# - the true and false branches.
class Node:
    def __init__(self, column=-1, value=None, true_branch=None, false_branch=None):
        #TODO: implement four attributes of the class Node
        pass

    # Below we defined a __repr__ method to ensure readable printing of Node instances
    def __repr__(self):
        return f'column: {self.column};\n' \
               f'value: {self.value};\n' \
               f'true branch: {self.true_branch};\n' \
               f'false branch: {self.false_branch}'
