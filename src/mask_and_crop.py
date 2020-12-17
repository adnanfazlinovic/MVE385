# Import libraries
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from skimage.transform import rotate  # test
import pathlib
import os
import argparse

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


def save_and_return_cropped(image_as_array, imagefolder_new, imagetype_new, imagename):
    """
    Saves and returns automatically cropped image.
    """

    # Check if there is a section of the row that is 90% non-white
    def half_row_nonwhite(row, verbose=False):
        nonwhite_in_a_row_ratio = 0
        white_in_a_row_ratio = 0
        for pixel in row:
            if np.sum(pixel) < 255 * row.shape[1]:
                nonwhite_in_a_row_ratio += 1 / row.shape[0]
            elif white_in_a_row_ratio < 0.05:
                nonwhite_in_a_row_ratio += 1 / row.shape[0]
                white_in_a_row_ratio += 1 / row.shape[0]
            else:
                nonwhite_in_a_row_ratio = 0
                white_in_a_row_ratio = 0
            if nonwhite_in_a_row_ratio > 0.5:
                return True
        return False

    # Calculate start and end row (or col)
    def start_and_end(image_as_array):
        start, end = 0, 0
        for r in range(image_as_array.shape[0]):
            sufficient_contrast = (
                np.abs(
                    int(np.sum(image_as_array[r])) - int(np.sum(image_as_array[r + 1]))
                )
                > 0.3 * 255 * image_as_array.shape[1] * image_as_array.shape[2]
            )
            if sufficient_contrast:
                if start == 0 and half_row_nonwhite(image_as_array[r + 1]):
                    start = r + 1
                    continue
                if start > 0 and half_row_nonwhite(image_as_array[r]):
                    end = r + 1
                    break
        return start, end

    y1, y2 = start_and_end(image_as_array)
    x1, x2 = start_and_end(np.swapaxes(image_as_array, 0, 1))
    cropped_image_as_array = image_as_array[y1:y2, x1:x2, :]

    # Save cropped image
    FILEPATH_cropped = imagefolder_new + imagename.split(".")[0] + imagetype_new
    Image.fromarray(cropped_image_as_array).save(FILEPATH_cropped)
    return cropped_image_as_array


def save_mask(image_as_array, imagefolder_new, imagetype_new, imagename):
    """
    Creates different masks, combines them and saves the resulting mask.
    """

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
    gray_scale_img = normalize_image(image_as_array)

    # Create mask for horizontal lines
    h_lines_threshold = args.h_lines_threshold
    FILEPATH_mask = imagefolder_new + imagename + "_mask" + imagetype_new
    mask_horizontal_lines = find_horizontal_lines(gray_scale_img, h_lines_threshold)

    # Create mask for vertical lines
    v_lines_threshold = args.v_lines_threshold
    FILEPATH_mask = imagefolder_new + imagename.split(".")[0] + "_mask" + imagetype_new
    mask_vertical_lines = find_vertical_lines(gray_scale_img, v_lines_threshold)

    # Create mask for beamstop
    beamstop_threshold = args.beamstop_threshold
    coordinates = select_area_of_interest(gray_scale_img)
    mask_beamstop = create_mask(gray_scale_img, coordinates, beamstop_threshold)

    # Create mask for vertical line (holder)
    val = input("Is there a beamstop holder to mask ([Y]/n)? ")
    if val == "Y" or val == "":
        mask_vert = select_vert(gray_scale_img)

        # Combine masks
        mask_as_array = (
            mask_horizontal_lines * mask_vertical_lines * mask_beamstop * mask_vert
        )
    # Combine masks
    else:
        mask_as_array = mask_horizontal_lines * mask_vertical_lines * mask_beamstop

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


if __name__ == "__main__":
    # Define paths and image types
    imagefolder = str(pathlib.Path(__file__).resolve().parents[1]) + "/Data/Raw data/"
    imagefolder_new = (
        str(pathlib.Path(__file__).resolve().parents[1]) + "/Data/Input data/"
    )
    imagetype = ".eps"
    imagetype_new = ".png"

    images = os.listdir(imagefolder)
    images = [
        image for image in images if "fig" not in image and ".DS_Store" not in image
    ]
    # Mask and crop all images in the Raw data folder
    for i, imagename in enumerate(images):

        FILEPATH_img = imagefolder + imagename

        # Read image
        image_as_array = plt.imread(FILEPATH_img)

        # image_as_array = (rotate(image_as_array,90,resize=True)*255).astype(np.uint8) # run with -vl 0.01, -hl 0.005. Testing vertical lines fun
        try:
            image_as_array = save_and_return_cropped(
                image_as_array, imagefolder_new, imagetype_new, imagename
            )
            print(
                "(",
                i + 1,
                "/",
                len(images),
                ") successfully cropped ",
                imagename,
                sep="",
            )
        except IndexError as error:
            print(
                "(",
                i + 1,
                "/",
                len(images),
                ") failed to crop ",
                imagename,
                "\t\t\t\t\t\t error: ",
                error,
                sep="",
            )

        save_mask(image_as_array, imagefolder_new, imagetype_new, imagename)