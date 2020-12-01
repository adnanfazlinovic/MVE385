# Import libraries
import numpy as np
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
import pathlib


# Read in image
#imagefolder = str(pathlib.Path(__file__).resolve().parents[1]) + "/Data/Input_data/"
#img = plt.imread(imagefolder + "MD2_MV_bulk_2_new.png")


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


def select_area_of_interest(img):

    """
    User-friendly approach which lets the user manually select the area corresponding to the beam stop.
    If not correctly selected the first time, the user can re-do this procedure.
    Close plots manually to continue in the procedure.
    Type Y for 'yes' or N for 'no' when deciding if the area is good.
    NOTE: SELECT UPPER-LEFT CORNER FIRST, THEN LOWER-RIGHT CORNER!
    """

    # Local variable which breaks loop if area of interest is selected well
    OK = False

    # Main while-loop
    while OK == False:

        # Plot image
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, cmap="gray")

        # Let user specify points
        coord = np.asarray(plt.ginput(2, show_clicks=True))
        rec = patches.Rectangle(
            (int(coord[0, 0]), int(coord[0, 1])),
            int(coord[1, 0] - coord[0, 0]),
            int(coord[1, 1] - coord[0, 1]),
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        # Include area of interest in plot
        ax.add_patch(rec)
        plt.draw()
        plt.show()

        # Ask user to accept or reject the proposed area of interest
        val = input("Is the region correct ([Y]/n)?\n")

        # Break if OK, re-do if not
        if val == "Y" or val == "":
            OK = True

    # Returns the coordinates which the user selected
    return coord


def create_mask(img, points, tol):

    """
    Creates a mask which marks the beam stop based on the coordinates given by the user.
    img is the image that will be processed. Coordinates is output from previous method.
    tol includes pixels which might not be 0, but very small in the area of interest.
    """

    # Split the coordinates given (numpy array) into two separate arrays, corresponding to each click.
    c_1 = points[0]
    c_2 = points[1]

    # Find range for x and y coordinates
    x_start = int(points[0, 0])
    y_start = int(points[0, 1])
    x_end = int(points[1, 0])
    y_end = int(points[1, 1])

    # Create mesh which represents area of interest
    y_range = np.arange(x_start, x_end)
    x_range = np.arange(y_start, y_end)

    # Create mask
    mask = np.ones((img.shape[0], img.shape[1]))
    for x in x_range:
        for y in y_range:
            if img[x, y] < tol:
                mask[x, y] = 0

    # Return mask which is the area of interest with value 1, 0 else
    return mask


def compare_mask_and_img(img, mask):

    """
    Plots original image (grayscale) and mask side-by-side.
    Doesn't have to be executed. Just a sanity check.
    """

    # Create figure
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)

    # Plot grayscale image
    ax1.imshow(img, cmap="gray")

    # Plot mask
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(mask, cmap="gray")

    # Show plot
    plt.show()