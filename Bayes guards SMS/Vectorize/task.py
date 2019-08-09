import numpy as np


def split_by_words(X):
    X = np.core.chararray.lower(X)
    return np.core.chararray.split(X)


def vectorize(X):
    X_len = len(X)
    X = split_by_words(X)

    uniques = np.unique(np.hstack(X))
    index_dict = {}
    for index, word in enumerate(uniques):
        index_dict[word] = index

    vectorization = np.zeros((X_len, len(index_dict)), dtype=np.int64)
    for index, message in enumerate(X):
        unique, count = np.unique(message, return_counts=True)
        for i, word in enumerate(unique):
            word_index = index_dict[word]
            vectorization[index, word_index] = count[i]

    return vectorization, index_dict