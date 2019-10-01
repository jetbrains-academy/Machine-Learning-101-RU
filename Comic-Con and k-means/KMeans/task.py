from PIL import Image
import numpy as np


def read_image(path='superman-batman.png'):
    image = Image.open(path)
    return np.array(image).reshape(-1, 3)


def euclidean_distance(A, B):
    return np.sqrt(np.sum(np.square(A - B), axis=1))


def init_clusters(n_clusters, n_features):
    return np.random.random_integers(low=0, high=255, size=(n_clusters, n_features))


def k_means(X, n_clusters, distance_metric):
    n_samples, n_features = X.shape
    classification = np.zeros(n_samples)
    clusters = init_clusters(n_clusters, n_features)
    distance = np.zeros((n_clusters, n_samples))

    while True:
        for i, c in enumerate(clusters):
            distance[i] = distance_metric(X, c)
        new_classification = np.argmin(distance, axis=0)
        if np.sum(new_classification != classification) == 0:
            break
        classification = new_classification
        for i in range(n_clusters):
            mask = classification == i
            clusters[i] = np.sum(X[mask], axis=0) / np.sum(mask)
    return classification, clusters


if __name__ == '__main__':
    image = read_image()
    (centroids, labels) = k_means(image, 4, euclidean_distance)
    print("Cluster centers:")
    for label in labels:
        print(label)
