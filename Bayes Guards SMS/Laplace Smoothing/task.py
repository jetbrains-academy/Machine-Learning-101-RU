import numpy as np
import codecs
import string


def test_train_split(X, y, ratio=0.8):
    mask = np.random.uniform(size=len(y)) < ratio
    return X[mask], y[mask], X[~mask], y[~mask]


def split_by_words(X):
    return np.core.chararray.lower(X).translate(str.maketrans('', '', string.punctuation)).split()


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

    return index_dict, vectorization


class NaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        self.unique_classes = np.unique(y)

        self.dictionary, X = vectorize(X)
        self.dict_size = len(self.dictionary)

        self.classes_prior = np.zeros(len(self.unique_classes), dtype=np.float64)
        self.classes_words_count = np.zeros(len(self.unique_classes), dtype=np.float64)
        self.likelihood = np.full((len(self.unique_classes), self.dict_size + 1), self.alpha, dtype=np.float64)

        for i, clazz in enumerate(self.unique_classes):
            y_i_mask = y == clazz
            y_i_sum = np.sum(y_i_mask)
            self.classes_prior[i] = y_i_sum / len(y)
            self.classes_words_count[i] = np.sum(X[y_i_mask])
            self.likelihood[i, :-1] += np.sum(X[y_i_mask], 0)

            denominator = self.classes_words_count[i] + self.alpha * self.dict_size
            self.likelihood[i] = self.likelihood[i] / denominator


def read_data(path):
    file = codecs.open(path, encoding='latin1')
    text = np.loadtxt(file, dtype=np.bytes_, delimiter='\t', unpack=True)
    return np.core.chararray.decode(text)


if __name__ == '__main__':
    y, X = read_data('spam.txt')
    X_train, y_train, X_test, y_test = test_train_split(X, y)

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    print(nb.classes_words_count)
    print(nb.classes_prior)
    print(nb.likelihood)
