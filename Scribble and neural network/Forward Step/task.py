import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(fname):
    mnist = pd.read_csv(fname)
    y = np.array(mnist.iloc[:, 0])
    X = np.array(mnist.iloc[:, 1:])
    return X, y


def show_images(X):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(X[i].reshape(28, -1), cmap="gray")
        plt.xticks([])
        plt.yticks([])
    plt.show()
    plt.close()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]

    def forward(self, X):
        for b, w in zip(self.biases, self.weights):
            X = sigmoid(np.dot(w, X) + b)
        return X


if __name__ == '__main__':
    X, y = read_data('../mnist.csv')
    show_images(X[0:9])
