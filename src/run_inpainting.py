from inpainting import *
import argparse
import pathlib


def test_hyperparameters_advanced(image_mask_pairs):
    global args
    lrs = [0.0001, 0.001, 0.01, 0.05, 0.1]
    param_noises = [False, True]
    reg_noise_stds = [0.3, 0.03, 0.003, 0]
    for image_mask_pair in image_mask_pairs:
        print("----------------", image_mask_pair[0], "----------------")
        for lr in lrs:
            for param_noise in param_noises:
                for reg_noise_std in reg_noise_stds:
                    inp = Inpainting(
                        args,
                        image_mask_pair[0],
                        image_mask_pair[1],
                        "vase",
                        lr=lr,
                        param_noise=param_noise,
                        reg_noise_std=reg_noise_std,
                    )
                    inp.perform_inpainting()


def test_hyperparameters_basic(image_mask_pairs):
    global args
    vase_and_kate_and_library = ["vase", "kate", "library"]
    nets = ["skip_depth6", "UNET", "ResNet", "skip_depth3", "skip_depth12"]
    for image_mask_pair in image_mask_pairs:
        print("----------------", image_mask_pair[0], "----------------")
        for vase_or_kate_or_library in vase_and_kate_and_library:
            if vase_or_kate_or_library == "library":
                for net in nets:
                    print("Now processing: ", vase_or_kate_or_library, "with net", net)
                    inp = Inpainting(
                        args,
                        image_mask_pair[0],
                        image_mask_pair[1],
                        vase_or_kate_or_library,
                        net,
                    )
                    inp.perform_inpainting()
            else:
                print("Now processing: ", vase_or_kate_or_library)
                inp = Inpainting(
                    args,
                    image_mask_pair[0],
                    image_mask_pair[1],
                    vase_or_kate_or_library,
                )
                inp.perform_inpainting()


def inpaint_image(image_mask_pairs):
    global args
    inp = Inpainting(args, image_mask_pair[0], image_mask_pair[1], "vase")
    inp.perform_inpainting()


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--plot",
    help="Plots input image, mask and the mask applied to the input image",
    action="store_true",
)
parser.add_argument(
    "--save_every",
    help="How many iterations between every save.",
    default=100,
    type=int,
)
parser.add_argument("--num_iter", help="Number of iterations.", default=500, type=int)
parser.add_argument(
    "--tuning",
    help="Either test different models (basic), or perform hyperparameter optimization (advanced).",
    type=str,
)
args = parser.parse_args()

# Input images and masks
imagefolder = str(pathlib.Path(__file__).resolve().parents[1]) + "/Data/Input data/"
files = os.listdir(imagefolder)
masks = []
for f in files:
    if "mask" in f:
        masks.append(f)
        files = [ff for ff in files if ff != f]  # remove f from files

image_mask_pairs = []
for m in masks:
    for f in files:  # files now only contain images, and not masks
        if m.split("_mask")[0] == f.split(".")[0]:
            image_mask_pairs.append((f, m))

if args.tuning == "basic":
    test_hyperparameters_basic(image_mask_pairs)
if args.tuning == "advanced":
    test_hyperparameters_advanced(image_mask_pairs)
else:
    inpaint_image()