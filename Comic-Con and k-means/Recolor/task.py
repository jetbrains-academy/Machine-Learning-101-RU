from PIL import Image
from PIL import ImageDraw
import numpy as np

IMAGE_WIDTH = 768
IMAGE_HEIGHT = 1024

def read_image(path='superman-batman.png'):
    image = Image.open(path)
    return np.array(image).reshape(-1, 3)


def euclidean_distance(A, B):
    return np.sqrt(np.sum(np.square(A - B), axis=1))


def k_means(X, n_clusters, distance_metric):
    n_samples, n_features = X.shape
    classification = np.zeros(n_samples)
    clusters = np.random.random_integers(low=0, high=255, size=(n_clusters, n_features))
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
            total_classified = np.sum(mask)
            is_empty_cluster = total_classified == 0
            if not is_empty_cluster:
                clusters[i] = np.sum(X[mask], axis=0) / total_classified
            else:
                clusters[i] = X[np.argmax(distance[i])]
    return classification, clusters


def centroid_histogram(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return counts


def plot_colors(hist, centroids):
    bar = np.zeros((50, 500, 3), dtype=np.uint8)
    hist_plot = Image.fromarray(bar)
    draw = ImageDraw.ImageDraw(hist_plot)
    start_x = 0
    sum_hist = np.sum(hist)
    for (percent, color) in zip(hist, centroids):
        end_x = start_x + percent * 500 / sum_hist
        draw.rectangle(((int(start_x), 0), (int(end_x), 50)), tuple(color))
        start_x = end_x

    hist_plot.save("histogram.png")


def recolor(image, n_colors):
    (labels, centroids) = k_means(image.astype(np.int64), n_colors, euclidean_distance)
    return centroids[labels]


if __name__ == '__main__':
    image = read_image()
    recolored_image = recolor(image, 4).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3).astype('uint8')
    image = Image.fromarray(recolored_image)
    image.save("recolored-superman-batman.png")
