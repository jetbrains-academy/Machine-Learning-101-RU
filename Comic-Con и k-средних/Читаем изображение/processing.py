import numpy as np
from PIL import Image
from clustering import k_means
from distances import euclidean_distance
from task import read_image


IMAGE_WIDTH = 768
IMAGE_HEIGHT = 1024


def recolor(image, n_colors):
    # Here we bring all together, by calling k_means on the image with a number
    # of colors to recolor it into provided
    # Note, that the image array is better unified by calling .astype(np.int64) on it
    imageint64 = image.astype(np.int64)
    (labels, centroids) = #TODO
    return centroids[labels]


# Call this function inside the main method in task.py to recolor the image!
def process_image():
    image = read_image("superman-batman.png")
    recolored_image = recolor(image, 8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3).astype('uint8')
    image = Image.fromarray(recolored_image)
    image.save("recolored-superman-batman.png")
