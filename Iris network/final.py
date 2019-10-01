import numpy as np
import pandas as pd


def read_data(fpath):
    iris = pd.read_csv(fpath)
    iris.loc[iris['species'] == 'virginica', 'species'] = 0
    iris.loc[iris['species'] == 'versicolor', 'species'] = 1
    iris.loc[iris['species'] == 'setosa', 'species'] = 2
    iris = iris[iris['species'] != 2]
    return iris[['petal_length', 'petal_width']].values, iris[['species']].values.astype('uint8')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NN:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = 2 * np.random.random((input_size, hidden_size)) - 1
        self.w2 = 2 * np.random.random((hidden_size, output_size)) - 1

    def feedforward(self, X):
        # Calculating the values of Nodes present in hidden layer using input layer and weight matrix (w1)
        self.layer1 = sigmoid(np.dot(X, self.w1))
        # Calculating the value of output nodes using hidden layer nodes and weight matrix (w2)
        l2 = sigmoid(np.dot(self.layer1, self.w2))
        return l2

    def train(self, X, y, n_iter=200000):
        for itr in range(n_iter):
            l2 = self.feedforward(X)
            self.backward(X, y, l2)

    def backward(self, X, y, l2, learning_rate=0.01):
        l2_delta = (y - l2) * sigmoid_derivative(l2)
        l1_delta = np.dot(l2_delta, self.w2.T) * sigmoid_derivative(self.layer1)
        self.w2 += (np.dot(self.layer1.T, l2_delta) * learning_rate)
        self.w1 += (np.dot(X.T, l1_delta) * learning_rate)

    def predict(self, X):
        return self.feedforward(X)


def train_test_split(X, y, ratio=0.8):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    train_len = int(X.shape[0] * ratio)
    return X[indices[:train_len]], y[indices[:train_len]], X[indices[train_len:]], y[indices[train_len:]]


# def evaluate(nn, X_test, y_test):
#     nn_y = nn.predict(X_test)
#     return (abs(y_test - nn_y)).mean()

def evaluate(nn, X_test, y_test):
    nn_y = nn.predict(X_test)
    return ((nn_y > 0.5).astype(int) == y_test).sum() / len(y_test)

if __name__ == '__main__':
    X, y = read_data('./Forward Step/iris.csv')
    trainX, trainY, testX, testY = train_test_split(X, y, 0.7)
    nn = NN(len(X[0]), 5, 1)
    nn.train(trainX, trainY)
    print(evaluate(nn, testX, testY))
