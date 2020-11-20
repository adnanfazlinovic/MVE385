# Import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2

#%% # Read image

im_path = "./Raw data/MD2_MV_bulk_2.eps"  #"./Raw data/MD2_MV_shear_2.eps" #"./Raw data/MD2_MV_edge_2.eps"  
image = plt.imread(im_path)
plt.imshow(image)

#%% Crop image

img =  image[39:265, 47:329, :] # Manual crop for MD2_MV_**_2 
plt.imshow(img)

#%%
(height, width, depth) = img.shape # Get dimensions of image

# Convert to grayscale then binary
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, bw_img = cv2.threshold(img_grayscale,9,1,cv2.THRESH_BINARY)
cv2.imshow("Binary Image", bw_img*255)

#%% Apply horizontal averaging
# kval1 = 1
# kval2 = width
# kernel = np.ones((kval1,kval2),np.float32)/(kval1*kval2)
# bw_img_av = cv2.filter2D(bw_img*255,-1,kernel)

bw_img_av = np.transpose(np.tile(bw_img.sum(axis=1)/width,(width,1)))
cv2.imshow("Averaged Image", bw_img_av)

#%% Convert to binary again
ret, bw_img2 = cv2.threshold(bw_img_av,0.01,1,cv2.THRESH_BINARY)
cv2.imshow("Binary Image 2", bw_img2*255)

#%%

# Check if we get same lines as manually
im_path = "./Our_data/tetra2_mask.png"
image_mask_man = plt.imread(im_path)
cv2.imshow("Difference", abs(image_mask_man[:,:,0]-bw_img2)*255)


