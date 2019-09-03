import numpy as np


def read_data(fname):
    data = np.genfromtxt(fname, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # rescale features
    X = np.concatenate((-np.ones(len(X)).reshape((-1, 1)), X), axis=1)
    y = -(y * 2 - 1)  # {0, 1} -> {1, -1}
    return X, y


def log_loss(M):
    return np.log2(1 + np.e**(-M)), -1 / (1 + np.e**M)


def sigmoid_loss(M):
    return 2 / (1 + np.exp(M)), -2 * np.exp(M) / (np.exp(M) + 1) ** 2


class GradientDescent:
    def __init__(self, *, alpha, threshold=1e-2, loss=sigmoid_loss):
        self.weights = []
        self.alpha = alpha
        self.threshold = threshold
        self.loss = loss

    def fit(self, X, y):
        n = X.shape[1]
        self.weights = np.random.uniform(-1 / (2 * n), 1 / (2 * n), size=n)
        errors = []

        while True:
            M = X.dot(self.weights) * y
            loss, derivative = self.loss(M)

            grad_q = np.sum((derivative.T * (X.T * y)).T, axis=0)

            tmp = self.weights - self.alpha * grad_q

            errors.append(np.sum(loss))
            if np.linalg.norm(self.weights - tmp) < self.threshold:
                break
            self.weights = tmp
        return errors

    def predict(self, X):
        return np.sign(X.dot(self.weights))


if __name__ == '__main__':
    X, y = read_data("pima-indians-diabetes.csv")
