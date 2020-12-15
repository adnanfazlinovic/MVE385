# Import libraries
import numpy as np
from matplotlib.patches import Polygon
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
import pathlib

# Read in image
#imagefolder = str(pathlib.Path(__file__).resolve().parents[1]) + "/Data/Input_data/"
#img = plt.imread(imagefolder + "MD2_MV_bulk_2_new.png")

def select_vert(img):

    """
    User-friendly approach which lets the user manually select the area corresponding to the verticle line.
    If not correctly selected the first time, the user can re-do this procedure.
    Close plots manually to continue in the procedure.
    Type Y for 'yes' or N for 'no' when deciding if the area is good.
    """

    # Local variable which breaks loop if area of interest is selected well
    OK = False

    # Main while-loop
    while OK == False:

        # Plot image
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, cmap="gray")

        # Let user specify points
        coord = np.asarray(plt.ginput(4, show_clicks=True))
        p = Polygon(coord, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_artist(p)
        # Include area of interest in plot
        plt.draw()
        plt.show()

        # Ask user to accept or reject the proposed area of interest
        val = input("Is the region correct ([Y]/n)?\n")

        # Break if OK, re-do if not
        if val == "Y" or val == "":
            OK = True

    """
    Creates a mask which marks the vertical line based on the coordinates given by the user.
    """
    
    x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='xy')
    x, y = x.flatten(), y.flatten()
    pts = np.vstack((x,y)).T
    pts_t = tuple(map(tuple, pts))
    mask = np.ones((img.shape[0],img.shape[1]))
    for (x,y) in pts_t:
        if p.get_path().contains_point((x,y)):
            mask[y][x] = 0

    # Return mask which is the area of interest with value 1, 0 else
    return mask


'''
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

if __name__ == "__main__":
    p = select_vert(img)
    mask = create_mask(img, p)
    compare_mask_and_img(img, mask)
'''