import codecs
from vectorize import *
from bayes import NaiveBayes


def test_train_split(X, y, ratio=0.8):
    mask = np.random.uniform(size=len(y)) < ratio
    return X[mask], y[mask], X[~mask], y[~mask]


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