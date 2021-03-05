import numpy as np


def init_clusters(n_clusters, n_features):
    # Here we create n initial clusters for a sample of objects with n features
    # The clusters are generated randomly among the acceptable values
    # In our case those are 0..255, for the RGB matrix
    return np.random.random_integers(low=0, high=255, size=(n_clusters, n_features))


def k_means(X, n_clusters, distance_metric):
    # The number of samples provided and the features involved are
    # equal to the dimensions of the sample matrix, as it has the
    # objects along one of the sides, and the features along the other
    n_samples, n_features = X.shape
    # The initial classification for each of the samples is a zero
    # array, as they are not yet affiliated with a class
    classification = np.zeros(n_samples)
    # Initializing the requested amount of clusters to adjust
    clusters = init_clusters(n_clusters, n_features)
    # For each of the sample we set the initial distance to all of the
    # clusters to zero, as the real distances are not yet measured
    distance = np.zeros((n_clusters, n_samples))

    # The algorithm will iterate until a stop condition is met
    while True:
        # We enumerate clusters and calculate the distances from all
        # of the samples to each of them
        for i, c in enumerate(clusters):
            distance[i] = #TODO
        # The assigned classes would be those corresponding to the nearest
        # cluster for each of the samples.
        # The new_classification is an array storing all of the
        # classes assigned
        new_classification = #TODO
        # The first stop condition would be met if new_classification
        # is exact the same as the classification (the previous one)
        if #TODO:
            break
        classification = new_classification
        # The following loop adjusts the centers of the clusters
        # Based on the mean values of the samples classified
        # E.g., if a cluster has two objects, (0, 0, 0) and (2, 2, 2)
        # its new center will be at (1, 1, 1)
        for i in range(n_clusters):
            mask = classification == i
            total_classified = np.sum(mask)
            # Here we also check if an empty cluster appears -
            # the one without any corresponding samples
            # Those could be dangerous and destabilize the
            # algorithm, resulting in it never finishing its
            # work. In case they occur, we assign a new center
            # to them, equal to a most distant sample
            is_empty_cluster = total_classified == 0
            if not is_empty_cluster:
                clusters[i] = #TODO
            else:
                clusters[i] = X[np.argmax(distance[i])]
    return classification, clusters
