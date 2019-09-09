import numpy as np
from PIL import Image


def read_data(fname):
    pass


def scale_to_unit_interval(ndar, eps=1e-8):
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]
    H, W = img_shape
    Hs, Ws = tile_spacing

    dt = X.dtype
    if output_pixel_vals:
        dt = 'uint8'
    out_array = np.zeros(out_shape, dtype=dt)

    for tile_row in range(tile_shape[0]):
        for tile_col in range(tile_shape[1]):
            if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                this_x = X[tile_row * tile_shape[1] + tile_col]
                if scale_rows_to_unit_interval:
                    this_img = scale_to_unit_interval(
                        this_x.reshape(img_shape))
                else:
                    this_img = this_x.reshape(img_shape)
                c = 1
                if output_pixel_vals:
                    c = 255
                out_array[
                tile_row * (H + Hs): tile_row * (H + Hs) + H,
                tile_col * (W + Ws): tile_col * (W + Ws) + W
                ] = this_img * c
    return out_array


def visualize_mnist(train_X):
    images = train_X[0:2500, :]
    image_data = tile_raster_images(images,
                                    img_shape=[28, 28],
                                    tile_shape=[50, 50],
                                    tile_spacing=(2, 2))
    im_new = Image.fromarray(np.uint8(image_data))
    im_new.save('mnist.png')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]

    def forward(self, X):
        for b, w in zip(self.biases, self.weights):
            X = sigmoid(np.dot(w, X) + b)
        return X
