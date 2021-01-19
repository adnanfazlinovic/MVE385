# Code for **"Inpainting"** figures $6$, $8$ and 7 (top) from the main paper.

"""
*Uncomment if running on colab* 
Set Runtime -> Change runtime type -> Under Hardware Accelerator select GPU in Google Colab 
"""
# !git clone https://github.com/DmitryUlyanov/deep-image-prior
# !mv deep-image-prior/* ./

# Import libs

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
from pathlib import Path
import argparse


from utils.inpainting_utils import *


class Inpainting:
    def __init__(
        self,
        args,
        image,
        mask,
        vase_or_kate_or_library,
        NET_TYPE="skip_depth6",
        lr=False,
        param_noise=False,
        reg_noise_std=False,
    ):
        self.args = args
        self.image = image
        self.mask = mask
        self.vase_or_kate_or_library = vase_or_kate_or_library
        self.NET_TYPE = NET_TYPE  # one of skip_depth4|skip_depth2|UNET|ResNet
        self.lr = lr
        self.param_noise = param_noise
        self.reg_noise_std = reg_noise_std

        self.i = 0

    def perform_inpainting(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32  # dtype = torch.cuda.FloatTensor

        PLOT = self.args.plot
        imsize = -1
        dim_div_by = 64

        # Path to input image, mask and output
        img_path = (
            str(Path(__file__).resolve().parents[1]) + "/data/Input data/" + self.image
        )
        mask_path = (
            str(Path(__file__).resolve().parents[1]) + "/data/Input data/" + self.mask
        )
        if self.args.tuning == "basic":
            if self.vase_or_kate_or_library == "library":
                folder = (
                    str(Path(__file__).resolve().parents[1])
                    + "/data/Output data/Hyperparameter optimization/Basic/"
                    + self.vase_or_kate_or_library
                    + "/"
                    + self.NET_TYPE
                    + "/"
                )
                Path(folder).mkdir(parents=True, exist_ok=True)
            else:
                folder = (
                    str(Path(__file__).resolve().parents[1])
                    + "/data/Output data/Hyperparameter optimization/Basic/"
                    + self.vase_or_kate_or_library
                    + "/"
                )
                Path(folder).mkdir(parents=True, exist_ok=True)
            outp_path = folder + "/plotout"
        elif self.args.tuning == "advanced":
            print(
                "lr =",
                self.lr,
                "param_noise =",
                self.param_noise,
                "reg_noise_std =",
                self.reg_noise_std,
            )
            folder = (
                str(Path(__file__).resolve().parents[1])
                + "/data/Output data/Hyperparameter optimization/Advanced/"
                + self.image
                + "/lr="
                + str(self.lr)
                + ", param_noise="
                + str(self.param_noise)
                + ", reg_noise_std="
                + str(self.reg_noise_std)
                + "/"
            )
            Path(folder).mkdir(parents=True, exist_ok=True)
            outp_path = folder + "/plotout"
        else:
            folder = (
                str(Path(__file__).resolve().parents[1])
                + "/data/Output data/"
                + self.image.split(".")[0]
                + "/"
            )
            Path(folder).mkdir(parents=True, exist_ok=True)
            outp_path = folder + "/plotout"

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
        num_iter = self.args.num_iter
        if self.args.tuning == "advanced":
            save_every = int(num_iter / 4)
        else:
            save_every = self.args.save_every

        if self.vase_or_kate_or_library == "vase":
            INPUT = "meshgrid"
            input_depth = 2
            LR = self.lr if self.lr else 0.01
            # num_iter = 5001
            param_noise = self.param_noise if self.param_noise else False
            # save_every = 50
            figsize = 32  # changed from 5
            reg_noise_std = self.reg_noise_std if self.reg_noise_std else 0.03

            net = (
                skip(
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
                )
                .type(dtype)
                .to(device)
            )

        elif self.vase_or_kate_or_library == "kate":
            # Same params and net as in super-resolution and denoising
            INPUT = "noise"
            input_depth = 32
            # num_iter = 6001
            LR = 0.01

            param_noise = False
            # save_every = 50
            figsize = 5
            reg_noise_std = 0.03

            net = (
                skip(
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
                )
                .type(dtype)
                .to(device)
            )

        elif self.vase_or_kate_or_library == "library":
            INPUT = "noise"
            input_depth = 1
            # num_iter = 3001
            # save_every = 50
            figsize = 8
            reg_noise_std = 0.00
            param_noise = True

            if "skip" in self.NET_TYPE:
                depth = int(self.NET_TYPE[-1])
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
                )

                LR = 0.01

            elif self.NET_TYPE == "UNET":

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

            elif self.NET_TYPE == "ResNet":

                net = ResNet(
                    input_depth,
                    img_np.shape[0],
                    8,
                    32,
                    need_sigmoid=True,
                    act_fun="LeakyReLU",
                )

                LR = 0.001
                param_noise = False

            else:
                assert False
        else:
            assert False

        net = net.type(dtype).to(device)
        net_input = (
            get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype).to(device)
        )

        # Compute number of parameters
        s = sum(np.prod(list(p.size())) for p in net.parameters())
        # print("Number of params: %d" % s)

        # Loss
        mse = torch.nn.MSELoss().type(dtype).to(device)

        img_var = np_to_torch(img_np).type(dtype).to(device)
        mask_var = np_to_torch(img_mask_np).type(dtype).to(device)

        # Main loop
        def closure():

            # global i

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
            if self.i % save_every == 0 or self.i == num_iter - 1:
                out_np = torch_to_np(out)
                out_np = 255 * np.moveaxis(out_np, 0, 2)
                out_np = out_np.astype(np.uint8)
                filep = outp_path + str(self.i) + ".png"
                image = Image.fromarray(out_np)
                image.save(filep)

            self.i += 1

            return total_loss

        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()

        p = get_params(OPT_OVER, net, net_input)
        optimize(OPTIMIZER, p, closure, LR, num_iter)