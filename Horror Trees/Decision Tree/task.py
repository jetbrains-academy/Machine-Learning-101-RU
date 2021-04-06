from node import Node                                                   # 1

import pandas as pd                                                     # 2
from calculate_entropy import entropy

import numpy as np                                                      # 3
from divide import Predicate

from tree import DecisionTree                                           # 5


def read_data(path):                                                    # 2
    data = pd.read_csv(path)
    y = data[['type']]
    X = data.drop('type', 1)
    return X.to_numpy(), y, X.columns.values


if __name__ == '__main__':
    node = Node(1, 2, [1, 2], [3, 4])                                   # 1
    print(node)

    X, y, columns = read_data("halloween.csv")                          # 2
    print('dataset entropy:', entropy(y))

    predicate = Predicate(3, 'clear')                                   # 3
    X = np.array([[1, 1, 1, 'clear'],
                  [2, 2, 2, 'clear'],
                  [3, 3, 3, 'green'],
                  [1, 2, 3, 'black']])
    y = np.array([1, 2, 3, 4])

    X1, y1, X2, y2 = predicate.divide(X, y)
    print(f'Division result: '
          f'\nFirst group labels: {y1} '
          f'\nFirst group objects: {X1} '
          f'\nSecond group labels: {y2} '
          f'\nSecond group objects: {X2}')

    print(f'Information Gain: {predicate.information_gain(X, y)}')      # 4

    print(DecisionTree().build(X, y))

