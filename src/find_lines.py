import numpy as np
from skimage.color import rgb2gray
from PIL import Image


def find_horizontal_lines(imagename, img_as_array, imagefolder_new, imagetype_new):
    # Get dimensions of image
    (height, width, depth) = img_as_array.shape

    # Convert to grayscale and then binary
    img_grayscale = rgb2gray(img_as_array)
    img_bw = img_grayscale > 0.01  # manually chosen threshold 0.01

    # Apply horizontal averaging which creates grayscale, and then convert to binary
    img_av = np.transpose(np.tile(img_bw.sum(axis=1) / width, (width, 1)))
    img_bw = img_av > 0.01  # manually chosen threshold 0.01
    return img_bw


def find_vertical_lines():
    pass