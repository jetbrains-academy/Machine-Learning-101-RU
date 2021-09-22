import numpy as np
from loss_functions import sigmoid_loss


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
            # M /= np.max(np.abs(M), axis=0)
            loss, derivative = self.loss(M)

            grad_q = np.sum((derivative.T * (X.T * y)).T, axis=0)
            if np.linalg.norm(grad_q) > 100:
                grad_q *= (100.0/grad_q.max())
            # grad_q /= np.max(np.abs(grad_q), axis=0)
            # grad_q *= (1.0/grad_q.max())
            new_weights = self.weights - self.alpha * grad_q

            errors.append(np.sum(loss))
            if np.linalg.norm(self.weights - new_weights) < self.threshold:
                break
            self.weights = new_weights
        return errors

    def predict(self, X):
        return np.sign(X.dot(self.weights))