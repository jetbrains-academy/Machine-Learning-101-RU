import numpy as np


def train_test_split(X, y, ratio=0.8):
    indices = np.random.permutation(X.shape[0])
    train_len = int(X.shape[0] * ratio)
    return X[indices[:train_len]], y[indices[:train_len]], X[indices[train_len:]], y[indices[train_len:]]


def knn(X_train, y_train, X_test, k, dist):
    def classify_single(x):
        dists = [dist(x, i) for i in X_train]
        indexes = np.argpartition(dists, k)[:k]
        return np.argmax(np.bincount(y_train[indexes]))

    return [classify_single(x) for x in X_test]


def euclidean_dist(x, y):
    return np.linalg.norm(x - y)


def taxicab_dist(x, y):
    return np.abs(x - y).sum()


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


def loocv(X_train, y_train, dist):
    def loo(k):
        c = 0
        for i in range(len(X_train)):
            x_train_cur = np.vstack([X_train[:i], X_train[i + 1:]])
            y_train_cur = np.concatenate((y_train[:i], y_train[i + 1:]))
            if knn(x_train_cur, y_train_cur, X_train[i:i + 1], k, dist)[0] != y_train[i]:
                c += 1
        return c

    loos = list(map(loo, range(1, len(X_train) - 1)))
    return np.argmin(loos) + 1


if __name__ == '__main__':
    wines = np.genfromtxt('wine.csv', delimiter=',')

    X, y = wines[:, 1:], np.array(wines[:, 0], dtype=np.int32)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.6)
    y_euclidean_predicted = knn(X_train, y_train, X_test, 5, euclidean_dist)
    print_precision_recall(precision_recall(y_euclidean_predicted, y_test))

    euclidean_opt = loocv(X_train, y_train, euclidean_dist)
    taxicab_opt = loocv(X_train, y_train, taxicab_dist)

    print("optimal euclidian k = " + str(euclidean_opt))
    print("optimal taxicab k = " + str(taxicab_opt))
    y_euclidean_predicted = knn(X_train, y_train, X_test, euclidean_opt, euclidean_dist)
    print_precision_recall(precision_recall(y_euclidean_predicted, y_test))

    y_taxicab_predicted = knn(X_train, y_train, X_test, taxicab_opt, euclidean_dist)
    print_precision_recall(precision_recall(y_taxicab_predicted, y_test))
