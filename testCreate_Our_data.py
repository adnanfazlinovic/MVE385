import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as im
from PIL import Image 

im_path = "/home/ingrid/Documents/Git/MVE385/Raw data/MD2_MV_bulk_2.eps"

image = plt.imread(im_path)
plt.imshow(image)

image_cropped = image[39:265, 47:329, :]
plt.imshow(image_cropped)
(height, width, depth) = image_cropped.shape

image_mask = np.zeros((height, width, depth), dtype=np.uint8)
image_mask += 1
maskval1 = 149
maskval2 = 158
maskval3 = 30 
maskval4 = 39
image_mask[maskval1:maskval2,:,:] = 0
image_mask[maskval3:maskval4,:,:] = 0
#plt.imshow(image_cropped*image_mask)
plotim = image_cropped*image_mask
plotim[maskval1:maskval2,:,:] = 255
plotim[maskval3:maskval4,:,:] = 255
plt.imshow(plotim)



#%%
plt.imshow(image_mask*255)

#%%

im.imsave('./Our_data-ingrid/tetra1.png', image_cropped, format = 'png')
im.imsave('./Our_data-ingrid/tetra1_mask.png', image_mask*255,  format = 'png')

#%%

im_path = "./Our_data-ingrid/tetra1.png"


img = Image.open(im_path)
ar = np.array(img)
img2 = Image.fromarray(ar[:,:,0:3])
img2.save('./Our_data-ingrid/tetra2.png')


#%%

im_path = "./Our_data-ingrid/tetra1_mask.png"


img = Image.open(im_path)
ar = np.array(img)
img2 = Image.fromarray(ar[:,:,0:3])
img2.save('./Our_data-ingrid/tetra2_mask.png')

