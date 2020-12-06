# Code for **"Inpainting"** figures $6$, $8$ and 7 (top) from the main paper.

"""
*Uncomment if running on colab* 
Set Runtime -> Change runtime type -> Under Hardware Accelerator select GPU in Google Colab 
"""
# !git clone https://github.com/DmitryUlyanov/deep-image-prior
# !mv deep-image-prior/* ./

#%% Import libs

from __future__ import print_function
import matplotlib.pyplot as plt

#%matplotlib inline

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
import torch
import torch.optim

from PIL import Image  # to save images
import pathlib
import argparse


from utils.inpainting_utils import *

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--plot",
    help="Plots input image, mask and the mask applied to the input image",
    action="store_true",
)
parser.add_argument(
    "--save_every", help="How many iterations between every save.", default=25, type=int
)
parser.add_argument("--num_iter", help="Number of iterations.", default=500, type=int)
args = parser.parse_args()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.FloatTensor  # dtype = torch.cuda.FloatTensor


PLOT = args.plot
imsize = -1
dim_div_by = 64

# Path to input image, mask and output
imtype = "edge"
img_path = (
    str(pathlib.Path(__file__).resolve().parents[1])
    + "/Data/Input_data/MD2_MV_"
    + imtype
    + "_2_new.png"
)
mask_path = (
    str(pathlib.Path(__file__).resolve().parents[1])
    + "/Data/Input_data/MD2_MV_"
    + imtype
    + "_2_mask.png"
)
outp_path = (
    str(pathlib.Path(__file__).resolve().parents[1])
    + "/Data/Output_data/"
    + imtype
    + "/plotout"
)

# Choose net
NET_TYPE = "skip_depth6"  # one of skip_depth4|skip_depth2|UNET|ResNet

# Load mask
img_pil, img_np = get_image(img_path, imsize)
img_mask_pil, img_mask_np = get_image(mask_path, imsize)

# Center crop
img_mask_pil = crop_image(img_mask_pil, dim_div_by)
img_pil = crop_image(img_pil, dim_div_by)

img_np = pil_to_np(img_pil)
img_mask_np = pil_to_np(img_mask_pil)

# Visualize
if PLOT:
    plot_image_grid([img_np, img_mask_np, img_mask_np * img_np], 3, 11)

# Setup
pad = "reflection"  # 'zero'
OPT_OVER = "net"
OPTIMIZER = "adam"

if "vase.png" in img_path:
    INPUT = "meshgrid"
    input_depth = 2
    LR = 0.01
    num_iter = 5001
    param_noise = False
    save_every = 50
    figsize = 32  # changed from 5
    reg_noise_std = 0.03

    net = skip(
        input_depth,
        img_np.shape[0],
        num_channels_down=[128] * 5,
        num_channels_up=[128] * 5,
        num_channels_skip=[0] * 5,
        upsample_mode="nearest",
        filter_skip_size=1,
        filter_size_up=3,
        filter_size_down=3,
        need_sigmoid=True,
        need_bias=True,
        pad=pad,
        act_fun="LeakyReLU",
    ).type(dtype)

elif (
    ("kate.png" in img_path)
    or ("peppers.png" in img_path)
    or ("Input_data" in img_path)
):
    # Same params and net as in super-resolution and denoising
    INPUT = "noise"
    input_depth = 32
    LR = 0.01
    num_iter = args.num_iter  # 6001 originally
    param_noise = False
    save_every = args.save_every
    figsize = 5
    reg_noise_std = 0.03

    net = skip(
        input_depth,
        img_np.shape[0],
        num_channels_down=[128] * 5,
        num_channels_up=[128] * 5,
        num_channels_skip=[128] * 5,
        filter_size_up=3,
        filter_size_down=3,
        upsample_mode="nearest",
        filter_skip_size=1,
        need_sigmoid=True,
        need_bias=True,
        pad=pad,
        act_fun="LeakyReLU",
    ).type(dtype)

elif "library.png" in img_path:

    INPUT = "noise"
    input_depth = 1

    num_iter = 3001
    save_every = 50
    figsize = 8
    reg_noise_std = 0.00
    param_noise = True

    if "skip" in NET_TYPE:

        depth = int(NET_TYPE[-1])
        net = skip(
            input_depth,
            img_np.shape[0],
            num_channels_down=[16, 32, 64, 128, 128, 128][:depth],
            num_channels_up=[16, 32, 64, 128, 128, 128][:depth],
            num_channels_skip=[0, 0, 0, 0, 0, 0][:depth],
            filter_size_up=3,
            filter_size_down=5,
            filter_skip_size=1,
            upsample_mode="nearest",  # downsample_mode='avg',
            need1x1_up=False,
            need_sigmoid=True,
            need_bias=True,
            pad=pad,
            act_fun="LeakyReLU",
        ).type(dtype)

        LR = 0.01

    elif NET_TYPE == "UNET":

        net = UNet(
            num_input_channels=input_depth,
            num_output_channels=3,
            feature_scale=8,
            more_layers=1,
            concat_x=False,
            upsample_mode="deconv",
            pad="zero",
            norm_layer=torch.nn.InstanceNorm2d,
            need_sigmoid=True,
            need_bias=True,
        )

        LR = 0.001
        param_noise = False

    elif NET_TYPE == "ResNet":

        net = ResNet(
            input_depth, img_np.shape[0], 8, 32, need_sigmoid=True, act_fun="LeakyReLU"
        )

        LR = 0.001
        param_noise = False

    else:
        assert False
else:
    assert False

net = net.type(dtype)
net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)

# Compute number of parameters
s = sum(np.prod(list(p.size())) for p in net.parameters())
print("Number of params: %d" % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_var = np_to_torch(img_np).type(dtype)
mask_var = np_to_torch(img_mask_np).type(dtype)

# Main loop

i = 0


def closure():

    global i

    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n = n + n.detach().clone().normal_() * n.std() / 50

    net_input = net_input_saved
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)

    total_loss = mse(out * mask_var, img_var * mask_var)
    total_loss.backward()

    # print('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')
    if i % save_every == 0 or i == num_iter:
        out_np = torch_to_np(out)
        out_np = 255 * np.moveaxis(out_np, 0, 2)
        out_np = out_np.astype(np.uint8)
        filep = outp_path + str(i) + ".png"
        image = Image.fromarray(out_np)
        image.save(filep)

    i += 1

    return total_loss


net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)