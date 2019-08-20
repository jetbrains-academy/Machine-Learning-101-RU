import numpy as np


def train_test_split(X, y, ratio=0.8):
    indices = np.random.permutation(X.shape[0])
    train_len = int(X.shape[0] * ratio)
    return X[indices[:train_len]], y[indices[:train_len]], X[indices[train_len:]], y[indices[train_len:]]


def knn(X_train, y_train, X_test, k, dist):
    def classify_single(x):
        dists = [dist(x, i) for i in X_train]
        nei_indexes = np.argpartition(dists, k)[:k]
        return np.argmax(np.bincount(y_train[nei_indexes]))

    return [classify_single(x) for x in X_test]


def euclidean_dist(x, y):
    return np.linalg.norm(x - y)


def taxicab_dist(x, y):
    return np.abs(x - y).sum()


if __name__ == '__main__':
    wines = np.genfromtxt('wine.csv', delimiter=',')

    X, y = wines[:, 1:], np.array(wines[:, 0], dtype=np.int32)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.6)
    y_predicted = knn(X_train, y_train, X_test, 5, euclidean_dist)