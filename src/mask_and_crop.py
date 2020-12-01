# Import libraries
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pathlib
import argparse

# Import modules
from find_lines import *
from find_beamstop import *


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
        "-l",
        "--lines_threshold",
        help="Defines pixel threshold for find_lines.py.",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "-b",
        "--beamstop_threshold",
        help="Defines pixel threshold for find_beamstop.py.",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    # Create mask for horizontal lines
    lines_threshold = args.lines_threshold
    FILEPATH_mask = imagefolder_new + imagename + "_mask" + imagetype_new
    mask_horizontal_lines = find_horizontal_lines(
        imagename, img_as_array, imagefolder_new, imagetype_new, lines_threshold
    )

    # Create mask for beamstop
    beamstop_threshold = args.beamstop_threshold
    gray_scale_img = normalize_image(img)
    coordinates = select_area_of_interest(gray_scale_img)
    mask_beamstop = create_mask(gray_scale_img, coordinates, beamstop_threshold)

    # Combine masks
    mask_as_array = mask_horizontal_lines * mask_beamstop

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
imagefolder = str(pathlib.Path(__file__).parents[1]) + "/Data/Raw data/"
imagefolder_new = str(pathlib.Path(__file__).parents[1]) + "/Data/Input_data/"
imagename = "MD2_MV_bulk_2"
imagetype = ".eps"
imagetype_new = ".png"
FILEPATH_img = imagefolder + imagename + imagetype

# Read image
image = plt.imread(FILEPATH_img)

# Manual crop of MD2_MV_**_2
img_as_array = image[39:265, 47:329, :]

# Save cropped image and mask
save_cropped(image)
save_mask(imagename)