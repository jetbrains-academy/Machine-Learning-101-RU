import numpy as np
import string

def split_by_words(X):
    return np.core.chararray.lower(X).translate(str.maketrans('', '', string.punctuation)).split()


def vectorize(X):
    # get the number of input messages
    X_len = len(X)
    # get a vector of words out of each message
    X = # TODO

    # get a 1D array of unique words
    uniques = # TODO
    # create an index dictionary and fill it with unique words and their indices
    index_dict = {}
    for index, word in enumerate(uniques):
        # TODO

    # create an array of zeros with dimensions corresponding
    # to input data size and index_dict length
    vectorization = # TODO
    # each i'th line of the array contains in the j'th position a number x
    # which shows how many times the i'th word was encountered in the j'th message
    for index, message in enumerate(X):
        unique, count = # TODO
        for i, word in enumerate(unique):
            # TODO

    return index_dict, vectorization