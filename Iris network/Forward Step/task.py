import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data(fpath):
    iris = pd.read_csv(fpath)
    iris.loc[iris['species'] == 'virginica', 'species'] = 0
    iris.loc[iris['species'] == 'versicolor', 'species'] = 1
    iris.loc[iris['species'] == 'setosa', 'species'] = 2
    iris = iris[iris['species'] != 2]
    return iris[['petal_length', 'petal_width']].values.T, iris[['species']].values.T.astype('uint8')


def plot_data(X, y):
    plt.scatter(X[0, :], X[1, :], c=y[0, :], s=40, cmap=plt.cm.Spectral)
    plt.title("IRIS DATA | Blue - Versicolor, Red - Virginica ")
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.show()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # (3x1) weight matrix from hidden to output layer

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(X, self.W1))  # activation function
        return sigmoid(np.dot(self.layer1, self.W2))  # final activation function


if __name__ == '__main__':
    X, y = read_data('iris.csv')
    plot_data(X, y)