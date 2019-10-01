import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(fpath):
    iris = pd.read_csv(fpath)
    iris.loc[iris['species'] == 'virginica', 'species'] = 0
    iris.loc[iris['species'] == 'versicolor', 'species'] = 1
    iris.loc[iris['species'] == 'setosa', 'species'] = 2
    iris = iris[iris['species'] != 2]
    return iris[['petal_length', 'petal_width']].values, iris[['species']].values.astype('uint8')


def plot_data(X, y):
    plt.scatter(X[0, :], X[1, :], c=y[0, :], s=40, cmap=plt.cm.Spectral)
    plt.title("IRIS DATA | Blue - Versicolor, Red - Virginica ")
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.show()


def train_test_split(X, y, ratio=0.8):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    train_len = int(X.shape[0] * ratio)
    return X[indices[:train_len]], y[indices[:train_len]], X[indices[train_len:]], y[indices[train_len:]]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NN:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = 2 * np.random.random((input_size, hidden_size)) - 1
        self.w2 = 2 * np.random.random((hidden_size, output_size)) - 1

    def feedforward(self, X):
        self.layer1 = sigmoid(np.dot(X, self.w1))
        return sigmoid(np.dot(self.layer1, self.w2))

    def backward(self, X, y, output, learning_rate=0.01):
        l2_delta = (y - output) * sigmoid_derivative(output)
        l1_delta = np.dot(l2_delta, self.w2.T) * sigmoid_derivative(self.layer1)
        self.w2 += (np.dot(self.layer1.T, l2_delta) * learning_rate)
        self.w1 += (np.dot(X.T, l1_delta) * learning_rate)


if __name__ == '__main__':
    X, y = read_data('iris.csv')
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.7)
    nn = NN(len(X[0]), 5, 1)
    output = nn.feedforward(X_train)
    nn.backward(X_train, y_train, output)
