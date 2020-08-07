from PIL import Image
import numpy as np


def read_image(path='superman-batman.png'):
    image = Image.open(path)
    return np.array(image).reshape(-1, 3)


if __name__ == '__main__':
    image = read_image()
