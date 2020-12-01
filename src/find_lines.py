import numpy as np
from skimage.color import rgb2gray
from PIL import Image


def find_horizontal_lines(
        img_as_array, threshold
):
    # Get dimensions of image
    (height, width) = img_as_array.shape

    # Convert to binary
    img_bw = img_as_array > threshold  # manually chosen threshold (default 0.01)

    # Apply horizontal averaging which creates grayscale, and then convert to binary
    img_av = np.transpose(np.tile(img_bw.sum(axis=1) / width, (width, 1)))
    img_bw = img_av > threshold  # manually chosen threshold (default 0.01)
    return img_bw


def find_vertical_lines(
    img_as_array, threshold
):
    # Get dimensions of image
    (height, width) = img_as_array.shape

    # Convert binary
    img_bw = img_as_array > threshold  # manually chosen threshold (default 0.01)

    # Apply horizontal averaging which creates grayscale, and then convert to binary
    img_av = np.tile(img_bw.sum(axis=0) / height, (height, 1))
    img_bw = img_av > threshold  # manually chosen threshold (default 0.005)
    return img_bw