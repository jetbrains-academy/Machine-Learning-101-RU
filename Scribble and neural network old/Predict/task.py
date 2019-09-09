import numpy as np
from PIL import Image


def read_data(fname):
    pass


def train_test_split(X, y, ratio=0.8):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    train_len = int(X.shape[0] * ratio)
    return X[indices[:train_len]], y[indices[:train_len]], X[indices[train_len:]], y[indices[train_len:]]


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

    def backpropagation(self, X, y):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        activation = X
        activations = [X]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = (activations[-1] - y) * sigmoid_derivative(zs[-1])
        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, activations[-2].T)

        for layer_index in range(2, self.num_layers):
            z = zs[-layer_index]
            spv = sigmoid_derivative(z)
            delta = np.dot(self.weights[-layer_index + 1].T, delta) * spv

            gradient_b[-layer_index] = delta
            gradient_w[-layer_index] = np.dot(delta, activations[-layer_index - 1].T)

        return gradient_b, gradient_w

    def train(self, X, y, n_iter=100, learning_rate=1):
        X = X.reshape(X.shape + (1,))
        data = np.array(list(zip(X, y)))

        for j in range(n_iter):
            np.random.shuffle(data)

            gradient_b = [np.zeros(b.shape) for b in self.biases]
            gradient_w = [np.zeros(w.shape) for w in self.weights]

            for x, y in data:
                delta_gradient_b, delta_gradient_w = self.backpropagation(x, y)
                gradient_b = [gb + dgb for gb, dgb in zip(gradient_b, delta_gradient_b)]
                gradient_w = [gw + dgw for gw, dgw in zip(gradient_w, delta_gradient_w)]

            #: :type : list[numpy.core.multiarray.ndarray]
            self.weights = [w - (learning_rate / len(data)) * nw for w, nw in zip(self.weights, gradient_w)]

            #: :type : list[numpy.core.multiarray.ndarray]
            self.biases = [b - (learning_rate / len(data)) * nb for b, nb in zip(self.biases, gradient_b)]

    def predict(self, X):
        X = X.reshape(X.shape + (1,))
        result = [np.argmax(self.forward(x)) for x in X]
        return result


if __name__ == '__main__':
    dataset = read_data('mnist-original.mat')
    trainX, testX, trainY, testY = train_test_split(
        dataset['data'].T / 255.0, dataset['label'].squeeze().astype("int0"), 0.7)

    nn = NeuralNetwork([trainX.shape[1], 100, 10])
    nn.train(trainX, trainY)
    nn.predict(testY)
