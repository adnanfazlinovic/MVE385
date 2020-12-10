# Import libraries
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pathlib
import argparse
from skimage.color import rgb2gray
import pandas as pd
import seaborn as sns


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


def intensity(image, center_point = [131, 114], angles = [-0.2, 0.2], direction = 'R'):
    
    center_x, center_y = center_point
    img = normalize_image(image)
    h, w = img.shape
    vl, vu = angles
    ku = np.tan(vu)
    kl = np.tan(vl)

    
    if (direction == 'D'):
        vals = list(range(center_y, h))
        center_p = center_x
        center_s = center_y
    elif (direction == 'U'):
        vals = list(range(0, center_y))
        vals.reverse()
        center_p = center_x
        center_s = center_y
    elif (direction == 'L'):
        vals = list(range(0, center_x))
        vals.reverse()
        center_p = center_y
        center_s = center_x
    else:
        vals = list(range(center_x, w))
        center_p = center_y
        center_s = center_x
  
    s_vals = np.array(vals)
    
    #Lists to save data
    I = []
    p_val_us = []
    p_val_ls = []
    
    for s_val in s_vals:
    
        p_val_u = center_p + int((s_val-center_s)*ku)
        p_val_l = center_p + int((s_val-center_s)*kl)
        p_val_us.append(p_val_u)
        p_val_ls.append(p_val_l)
    
        p_val_range = list(range(min(p_val_l, p_val_u),max(p_val_l, p_val_u)+1))
        intensity = sum([img[p_val, s_val] for p_val in p_val_range])/len(p_val_range)
        I.append(intensity)
            
    if (direction == 'D' or direction == 'U'):
        indlistu = list(zip(s_vals,p_val_us))
        indlistl = list(zip(s_vals,p_val_ls))
    else:
        indlistu = list(zip(p_val_us,s_vals))
        indlistl = list(zip(p_val_ls,s_vals))

    img_ang = image.copy()
    for (y,x) in indlistu:
        img_ang[y,x] = 1
    
    for (y,x) in indlistl:
        img_ang[y,x] = 1    
    
    sns.set_style('white')
    g = plt.imshow(img_ang)
        
    return vals, I, g


#%%

# Define paths and image types
imagefolder = str(pathlib.Path(__file__).resolve().parents[1]) + "/Data/Input_data/"
imagename = "MD2_MV_shear_2_new" 
imagetype = ".png"
FILEPATH_img = imagefolder + imagename + imagetype

# Read image
image = plt.imread(FILEPATH_img)

#%%

# Find intensity
q, I, g = intensity(image, direction = 'U')


df = pd.DataFrame({'q': q, 'I': I})

sns.set_style('whitegrid')

fig, ax = plt.subplots(num = 2)
sns.scatterplot(x='q', y='I', data=df, s=20)
#lm = sns.lmplot(x='x', y='I', data=df, ci=None, order=5, truncate=True)



                 


