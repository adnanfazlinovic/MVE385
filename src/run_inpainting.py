from inpainting import *
import argparse


def test_hyperparameters_advanced():
    global args
    lrs = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    param_noises = [False, True]
    reg_noise_stds = [0.3, 0.03, 0.003]

    for lr in lrs:
        for param_noise in param_noises:
            for reg_noise_std in reg_noise_stds:
                inp = Inpainting(
                    args,
                    "vase",
                    lr=lr,
                    param_noise=param_noise,
                    reg_noise_std=reg_noise_std,
                )
                inp.perform_inpainting()


def test_hyperparameters_basic():
    global args
    vase_and_kate_and_library = ["vase", "kate", "library"]
    nets = ["skip_depth6", "UNET", "ResNet", "skip_depth3", "skip_depth12"]

    for vase_or_kate_or_library in vase_and_kate_and_library:
        if vase_or_kate_or_library == "library":
            for net in nets:
                print("Now processing: ", vase_or_kate_or_library, "with net", net)
                inp = Inpainting(args, vase_or_kate_or_library, net)
                inp.perform_inpainting()
        else:
            print("Now processing: ", vase_or_kate_or_library)
            inp = Inpainting(args, vase_or_kate_or_library)
            inp.perform_inpainting()


def inpaint_image():
    global args
    inp = Inpainting(args, "kate")
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
parser.add_argument(
    "--imtype", help="Which imagetype to inpaint, e.g. shear.", default="edge", type=str
)
args = parser.parse_args()

if args.tuning == "basic":
    test_hyperparameters_basic()
if args.tuning == "advanced":
    test_hyperparameters_advanced()
else:
    inpaint_image()