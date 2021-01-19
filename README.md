<p align="center">
  <img src="https://github.com/adnanfazlinovic/MVE385/blob/main/data/Input%20data/MD2_MV_bulk_3.png" width="350" title="hover text">
  <img src="https://github.com/adnanfazlinovic/MVE385/blob/main/data/Old%20results/Sorted%20results/MD2_MV_bulk_3.png/1%20Good/CHOSEN_lr%3D0.01%2C%20param_noise%3DTrue%2C%20reg_noise_std%3D0/CHOSEN_plotout4500.png" width="350" alt="accessibility text">
</p>

# Healing X-ray Scattering Images with the Deep Image Prior
MVE385 - Project course in mathematical and statistical modelling in collaboration with Tetra Pak

## Introduction
Background:\
Tetra  Pak®  have  started  to  use  the  experimental  techniques:  Small  and  Wide  Angle  X-ray Scattering  (SAXS  and  WAXS)  at  several  different  length  scales  in  both  laboratory  and  synchrotron X-ray sources to study the structure and material orientation of semicrystalline polymers used in polymer tops and opening devices.

Scope:\
The purpose of the student project work is to evaluate if it is possible to use Image analysis, AI and/or machine learning to heal such scattering plots, described above, in an automatic or semi-automatic  manner to enhance the interpretation and facilitate the post  processing  and  analysis of the information available and possible to extract from the scattering pattern.

Outcome: \
The goal is to develop an automatic procedure for “image” healing utilizing “intelligence” and symmetry etc. to enhance the scattering patterns from SAXS/WAXS measurements.

Methods:\
The Deep Image Prior. The implementation of the Deep Image Prior comes from this repo https://github.com/DmitryUlyanov/deep-image-prior. Disclaimer: We take no credit for the implementation of the Deep Image Prior in this repo, the full credit goes to the contributors to the aforementioned repo.

## User guide
### Dependencies
To run the code, some packages are needed. You can create a conda environment with the required dependencies by running
`conda env create -f environment.yml`. Some packages might be missing, in which case they should be installed with pip, e.g. scikit-image (if module skimage is not found), torch, torchvision and matplotlib.

### Pre-processing
The occluded images should be placed in the Raw data folder. To crop and mask these images, run
`mask_and_crop.py` which is located in the `src` folder, with or without flags. This will crop every image so that the frame, if existent, is removed and the the image is masked. The resulting images are placed in the Input data folder. The flags are used to set different thresholds related to identifying occlusions:
- `h_lines_threshold`: float. Defines pixel threshold for horizontal lines in find_lines.py. Default: 0.01.
- `v_lines_threshold`: float. Defines pixel threshold for vertical lines in find_lines.py. Default: 0.005.
- `beamstop_threshold`: int. Defines pixel threshold for find_beamstop.py. Default: 10.

One can use none, one or several of these flags, for example `python --h_lines_threshold 0.02 --beamstop_threshold 0.003`.

### Inpainting
When there is data in the Input data folder, one can proceed to perform the inpainting. This is done by running the `run_inpainting.py` script, with or without flags. Running without flags will inpaint all images in the Input data folder with the (currently known) best hyperparameters. Below are a list of flags and their meaning:
- `tuning`: basic/advanced. Performs either basic or advanced hyperparameter tuning. Basic tuning tests the different configurations vase, kate and library that uses different network architectures. Advanced tuning uses the best configuration, vase, and tunes the three hyperparameters `lr`, `param_noises` and `reg_noise_stds`. Default: None, in which case no tuning is performed. Example usage: `python run_inpainting.py --tuning advanced`
- `num_iter`: int. Number of iterations to run each training. Default: 500. Example usage: `python run_inpainting.py --num_iter 3000`.
- `save_every`: int. Frequency with which output images are saved. Default: 100. Example usage: `python run_inpainting.py --save_every 200`.
- `plot`. Whether to plot the input image, mask and the mask applied to the input image before training. Default: No. Example usage: `python run_inpainting.py --plot`.

As before, all flags are optional.
