from cv2 import imread
from numpy import ndarray


def read_image(filename: str) -> ndarray:
    """
    Returns opencv image of the file filename
    :param filename: name of the image to load
    :return: opencv image
    """
    image = imread(filename, -1)
    if image is None or image.data is None:
        raise FileNotFoundError("Unable to read file " + filename)
    else:
        return image
