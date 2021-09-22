import numpy as np


def log_loss(M):
    # return 2 + M ** 2, 2 * M
    return np.log2(1 + np.exp(-M)), np.clip(-1 / (np.log(2)*(1 + np.exp(M))), -1, 1)


def sigmoid_loss(M):
    # return 2 + M ** 2, 2 * M
    return 2 / (1 + np.exp(M)), np.clip(-2 * np.exp(M) / (np.exp(M) + 1) ** 2, -1, 1)
