import cv2


def read_image(path='superman-batman.png'):
    return cv2.imread(path).reshape(-1, 3)


if __name__ == '__main__':
    image = read_image()
