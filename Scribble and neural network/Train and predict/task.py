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


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


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

    def backpropagation(self, X, y):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        activation = X
        activations = [X]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = (activations[-1] - y) * sigmoid_derivative(zs[-1])
        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, activations[-2].T)

        for layer_index in range(2, self.num_layers):
            z = zs[-layer_index]
            spv = sigmoid_derivative(z)
            delta = np.dot(self.weights[-layer_index + 1].T, delta) * spv

            gradient_b[-layer_index] = delta
            gradient_w[-layer_index] = np.dot(delta, activations[-layer_index - 1].T)

        return gradient_b, gradient_w


if __name__ == '__main__':
    X, y = read_data('../mnist.csv')
    show_images(X[0:9])
