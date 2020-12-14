# Import libraries
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pathlib
import argparse
from skimage.transform import rotate # test

# Import modules
from find_lines import *
from find_beamstop import *
from find_vert import *


def normalize_image(img):

    """
    Takes input image (usually RGB) and transforms it into grayscale.
    Normalizes the inputs so that they are ints ranging from 0 to 255.
    0 corresponds to smallest value (might not be 0 in original) and 255 to the largest (might not be 255 in original).
    Doesn't matter, this is only to make the mask.
    """

    # Load image and convert to grayscale
    img = rgb2gray(img)

    # Normalize values, range 0 to 255
    img = (img - img.min()) / (img.max() - img.min())
    img *= 255

    # Make int values
    img = img.astype(int)

    # Return new image
    return img


def save_cropped(image):
    """
    Saves cropped image.
    """
    global img_as_array, imagefolder_new, imagetype_new

    # Save cropped image
    FILEPATH_cropped = imagefolder_new + imagename + "_new" + imagetype_new
    Image.fromarray(img_as_array).save(FILEPATH_cropped)


def save_mask(imagename):
    """
    Creates different masks, combines them and saves the resulting mask.
    """
    global img_as_array, imagefolder_new, imagetype_new

    # Create argument parser for the thresholds
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-hl",
        "--h_lines_threshold",
        help="Defines pixel threshold for horizontal lines in find_lines.py.",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "-vl",
        "--v_lines_threshold",
        help="Defines pixel threshold for vertical lines in find_lines.py.",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "-b",
        "--beamstop_threshold",
        help="Defines pixel threshold for find_beamstop.py.",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    # Convert image to grayscale
    gray_scale_img = normalize_image(img_as_array)

    # Create mask for horizontal lines
    h_lines_threshold = args.h_lines_threshold
    FILEPATH_mask = imagefolder_new + imagename + "_mask" + imagetype_new
    mask_horizontal_lines = find_horizontal_lines(
        gray_scale_img, h_lines_threshold
    )
    
    # Create mask for vertical lines
    v_lines_threshold = args.v_lines_threshold
    FILEPATH_mask = imagefolder_new + imagename + "_mask" + imagetype_new
    mask_vertical_lines = find_vertical_lines(
        gray_scale_img, v_lines_threshold
    )

    # Create mask for beamstop
    beamstop_threshold = args.beamstop_threshold
    coordinates = select_area_of_interest(gray_scale_img)
    mask_beamstop = create_mask(gray_scale_img, coordinates, beamstop_threshold)

    # Create mask for vertical line (holder)
    mask_vert = select_vert(gray_scale_img)

    # Combine masks
    mask_as_array = mask_horizontal_lines * mask_vertical_lines * mask_beamstop * mask_vert

    # Convert from array to 3-channel PIL image
    mask_as_array_as_dummy_rgb = np.transpose(
        np.array(
            [
                np.transpose(mask_as_array * 255),
                np.transpose(mask_as_array * 255),
                np.transpose(mask_as_array * 255),
            ]
        ).astype(dtype=np.uint8)
    )
    img_mask = Image.fromarray(mask_as_array_as_dummy_rgb)

    # Save mask
    img_mask.save(FILEPATH_mask)


# Define paths and image types
imagefolder = str(pathlib.Path(__file__).resolve().parents[1]) + "/Data/Raw data/"
imagefolder_new = str(pathlib.Path(__file__).resolve().parents[1]) + "/Data/Input_data/"
imagename = "MD2_MV_edge_2" 
imagetype = ".eps"
imagetype_new = ".png"
FILEPATH_img = imagefolder + imagename + imagetype

# Read image
image = plt.imread(FILEPATH_img)


# Manual crop of MD2_MV_**_2
img_as_array = image[39:265, 47:329, :]

#img_as_array = (rotate(img_as_array,90,resize=True)*255).astype(np.uint8) # run with -vl 0.01, -hl 0.005. Testing vertical lines fun

# Save cropped image and mask
save_cropped(image)
save_mask(imagename)