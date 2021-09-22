import numpy as np
import matplotlib.pyplot as plt
from utils import train_test_split
from stochastic_gradient_descent import StochasticGradientDescent
from precision_recall import precision_recall, print_precision_recall
from loss_functions import sigmoid_loss, log_loss


def read_data(fname):
    data = np.genfromtxt(fname, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # rescale features
    X = np.concatenate((-np.ones(len(X)).reshape((-1, 1)), X), axis=1)
    y = -(y * 2 - 1)  # {0, 1} -> {1, -1}
    return X, y


if __name__ == '__main__':
    X, y = read_data("pima-indians-diabetes.csv")
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.8)
    n_iter = 5000

    for loss in [sigmoid_loss, log_loss]:
        for k in [1, 10, 50]:
            plt.clf()
            for alpha, color in zip([1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                                    ["red", "blue", "green", "magenta", "yellow", "cyan"]):
                gd = StochasticGradientDescent(alpha=alpha, k=k, n_iter=n_iter)
                plt.plot(gd.fit(X_train, y_train), label=str(alpha), color=color, alpha=0.7,
                         linewidth=1)
                print("SGD({}, k={}, alpha={})".format(loss.__name__, k, alpha))
                print_precision_recall(precision_recall(gd.predict(X_test), y_test))
                print(gd.weights.tolist())
            plt.ylim((plt.ylim()[0], min(1.5, plt.ylim()[1])))
            plt.title("SGD({}, k={})".format(loss.__name__, k))
            plt.legend()
            plt.savefig("sdg-{}-{}.png".format(loss.__name__, k))