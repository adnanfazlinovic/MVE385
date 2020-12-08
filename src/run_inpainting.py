from inpainting import *
import argparse


def test_hyperparameters():
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
    default=25,
    type=int,
)
parser.add_argument("--num_iter", help="Number of iterations.", default=500, type=int)
parser.add_argument(
    "--hyperparam_test", help="Run hyperparameter testing.", action="store_true"
)
parser.add_argument(
    "--imtype", help="Which imagetype to inpaint, e.g. shear.", default="edge", type=str
)
args = parser.parse_args()

if args.hyperparam_test:
    test_hyperparameters()
else:
    inpaint_image()