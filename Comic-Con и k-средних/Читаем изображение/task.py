from PIL import Image
import numpy as np


def read_image(path='superman-batman.png'):
    # Here we need to read the image using the PIL function open
    image = #TODO
    # We reshape the image array returned into one with (M x N, 3)
    # shape
    return np.array(image).reshape(-1, 3)


if __name__ == '__main__':
    image = read_image()
    # Take a look at what does the image look like in a form of array
    print(image)
