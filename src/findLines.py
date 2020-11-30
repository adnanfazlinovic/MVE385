# Import librarie
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
import pathlib

# Read imag
imagefolder = str(pathlib.Path(__file__).parents[1]) + "/Data/Raw data/"
imagename = "MD2_MV_bulk_2"
imagetype = ".eps"
FILEPATH_img = imagefolder + imagename + imagetype
image = plt.imread(FILEPATH_img)

# Manual crop for MD2_MV_**_2
img_as_array = image[39:265, 47:329, :]

# Save cropped image
imagefolder_new = str(pathlib.Path(__file__).parents[1]) + "/Data/Input_data/"
imagetype_new = ".png"
FILEPATH_cropped = imagefolder_new + imagename + "_new" + imagetype_new
Image.fromarray(img_as_array).save(FILEPATH_cropped)

# Get dimensions of image
(height, width, depth) = img_as_array.shape

# Convert to grayscale and then binary
img_grayscale = rgb2gray(img_as_array)
img_bw = img_grayscale > 0.01  # manually chosen threshold 0.01

# Apply horizontal averaging which creates grayscale, and theb convert to binary
img_av = np.transpose(np.tile(img_bw.sum(axis=1) / width, (width, 1)))
img_bw = img_av > 0.01  # manually chosen threshold 0.01

# Save image mask
FILEPATH_mask = imagefolder_new + imagename + "_mask" + imagetype_new
dummyrgb = np.transpose(
    np.array(
        [
            np.transpose(img_bw * 255),
            np.transpose(img_bw * 255),
            np.transpose(img_bw * 255),
        ]
    ).astype(dtype=np.uint8)
)
img_dummyrgb = Image.fromarray(dummyrgb)
img_dummyrgb.save(FILEPATH_mask)