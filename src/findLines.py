# Import libraries
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image

#%% # Read image
im_path = "../Data/Raw data/"
imagename = "MD2_MV_bulk_2"
# imagename = "MD2_MV_shear_2"
# imagename = "MD2_MV_edge_2"
imagetype = ".eps"
FILEPATH_img = im_path + imagename + imagetype
image = plt.imread(FILEPATH_img)
# plt.imshow(image)

#%% Crop image
img = image[39:265, 47:329, :]  # Manual crop for MD2_MV_**_2
# plt.imshow(img)

#%%
(height, width, depth) = img.shape  # Get dimensions of image

# Convert to grayscale then binary
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, bw_img = cv2.threshold(img_grayscale, 9, 1, cv2.THRESH_BINARY)
# cv2.imshow("Binary Image", bw_img*255)

#%% Apply horizontal averaging
bw_img_av = np.transpose(np.tile(bw_img.sum(axis=1) / width, (width, 1)))
# cv2.imshow("Averaged Image", bw_img_av)

#%% Convert to binary again
ret, bw_img2 = cv2.threshold(bw_img_av, 0.01, 1, cv2.THRESH_BINARY)
# cv2.imshow("Binary Image 2", bw_img2*255)

#%% Check if we get same lines as manually
# cv2.imshow("Difference", abs(image_mask_man[:,:,0]-bw_img2)*255)

#%% Save images

# Cropped image
im_path_new = "../Data/Input_data/"
imagetype_new = ".png"

FILEPATH_img_new = im_path_new + imagename + "_new" + imagetype_new
imgI = Image.fromarray(img)
imgI.save(FILEPATH_img_new)

# Image mask
FILEPATH_mask = im_path_new + imagename + "_mask" + imagetype_new
dummyrgb = np.transpose(
    np.array(
        [
            np.transpose(bw_img2 * 255),
            np.transpose(bw_img2 * 255),
            np.transpose(bw_img2 * 255),
        ]
    ).astype(dtype=np.uint8)
)
dummyrgbI = Image.fromarray(dummyrgb)
dummyrgbI.save(FILEPATH_mask)