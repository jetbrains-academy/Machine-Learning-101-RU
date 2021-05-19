import numpy as np
import matplotlib.pyplot as plt


def read_data(fname):
    data = np.genfromtxt(fname, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # rescale features
    X = np.concatenate((-np.ones(len(X)).reshape((-1, 1)), X), axis=1)
    y = -(y * 2 - 1)  # {0, 1} -> {1, -1}
    return X, y


def train_test_split(X, y, ratio=0.8):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    train_len = int(X.shape[0] * ratio)
    return X[indices[:train_len]], y[indices[:train_len]], X[indices[train_len:]], y[indices[train_len:]]


def precision_recall(y_pred, y_test):
    class_precision_recall = []
    for c in np.unique(y_test):
        tp = len([i for i in range(len(y_pred)) if y_pred[i] == c and y_test[i] == c])
        fp = len([i for i in range(len(y_pred)) if y_pred[i] == c and y_test[i] != c])
        fn = len([i for i in range(len(y_pred)) if y_pred[i] != y_test[i] and y_pred[i] != c])
        precision = tp / (tp + fp) if tp + fp > 0 else 0.
        recall = tp / (tp + fn) if tp + fn > 0 else 0.
        class_precision_recall.append((c, precision, recall))
    return class_precision_recall


def print_precision_recall(result):
    for c, precision, recall in result:
        print("class:", c, "\nprecision:", precision, "\nrecall:", recall, "\n")

def log_loss(M):
    return np.log2(1 + np.exp(M)), -1 / (1 + np.exp(M))


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
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.8)

    for loss in [sigmoid_loss, log_loss]:
        plt.clf()
        for alpha, color in zip([1e-6, 1e-4, 1e-2, 1], ["red", "blue", "green", "magenta"]):
            gd = GradientDescent(alpha=alpha, loss=loss, threshold=1e-5)
            plt.plot(gd.fit(X_train, y_train), label=str(alpha), color=color, alpha=0.7, linewidth=1)
            print("GradientDescent({}, alpha={})".format(loss.__name__, alpha))
            print_precision_recall(precision_recall(gd.predict(X_test), y_test))
            print(gd.weights.tolist())
        plt.ylim((plt.ylim()[0], min(1.5, plt.ylim()[1])))
        plt.title("GradientDescent({})".format(loss.__name__))
        plt.legend()
        plt.savefig("gradient-{}.png".format(loss.__name__))