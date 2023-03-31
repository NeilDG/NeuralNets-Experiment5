import itertools
import sys
from optparse import OptionParser
import random
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from config.network_config import ConfigHolder
from loaders import dataset_loader
import global_config
from utils import plot_utils
from testers import depth_tester
from tqdm import tqdm
from tqdm.auto import trange
from time import sleep
import yaml
from yaml.loader import SafeLoader

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--network_version', type=str, default="vXX.XX")
parser.add_option('--iteration', type=int, default=1)
parser.add_option('--plot_enabled', type=int, default=1)

def update_config(opts):
    global_config.server_config = opts.server_config
    global_config.plot_enabled = opts.plot_enabled
    global_config.general_config["network_version"] = opts.network_version
    global_config.general_config["iteration"] = opts.iteration
    network_config = ConfigHolder.getInstance().get_network_config()

    if (global_config.server_config == 0):  # COARE
        global_config.general_config["num_workers"] = 6
        global_config.disable_progress_bar = True
        global_config.path = "/scratch1/scratch2/neil.delgallego/SynthV3_Raw/{dataset_version}/sequence.0/"
        print("Using COARE configuration. Workers: ", global_config.general_config["num_workers"])

    elif (global_config.server_config == 1):  # CCS Cloud
        global_config.general_config["num_workers"] = 12
        global_config.path = "/home/jupyter-neil.delgallego/SynthV3_Raw/{dataset_version}/sequence.0/"
        print("Using CCS configuration. Workers: ", global_config.general_config["num_workers"])

    elif (global_config.server_config == 2):  # RTX 2080Ti
        global_config.general_config["num_workers"] = 6

        print("Using RTX 2080Ti configuration. Workers: ", global_config.general_config["num_workers"])

    elif (global_config.server_config == 3):
        global_config.general_config["num_workers"] = 12
        global_config.path = "X:/SynthV3_Raw/{dataset_version}/sequence.0/"
        print("Using RTX 3090 configuration. Workers: ", global_config.general_config["num_workers"])

    global_config.path = global_config.path.format(dataset_version=network_config["dataset_version"])
    global_config.exr_path = global_config.path + "*.exr"
    global_config.rgb_path = global_config.path + "*.camera.png"


def main(argv):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    manualSeed = 0
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    (opts, args) = parser.parse_args(argv)
    yaml_config = "./hyperparam_tables/{network_version}.yaml"
    yaml_config = yaml_config.format(network_version=opts.network_version)
    hyperparam_path = "./hyperparam_tables/common_iter.yaml"
    with open(yaml_config) as f, open(hyperparam_path) as h:
        ConfigHolder.initialize(yaml.load(f, SafeLoader), yaml.load(h, SafeLoader))

    update_config(opts)
    print(opts)
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (global_config.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    network_config = ConfigHolder.getInstance().get_network_config()
    print(network_config)

    general_config = global_config.general_config
    print(general_config)

    rgb_path = global_config.rgb_path
    exr_path = global_config.exr_path

    print("Dataset path: ", global_config.path)

    plot_utils.VisdomReporter.initialize()
    global_config.general_config["test_size"] = 128
    synth_loader, dataset_count = dataset_loader.load_test_dataset(rgb_path, exr_path)
    dt = depth_tester.DepthTester(device)
    start_epoch = global_config.general_config["current_epoch"]
    print("---------------------------------------------------------------------------")
    print("Started synth test loop for mode: depth", " Set start epoch: ", start_epoch)
    print("---------------------------------------------------------------------------")

    # compute total progress
    steps = general_config["test_size"]
    needed_progress = int(dataset_count / steps) + 1
    current_progress = 0
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)

    with torch.no_grad():
        for i, (rgb_batch, depth_batch) in enumerate(synth_loader, 0):
            rgb_batch = rgb_batch.to(device)
            depth_batch = depth_batch.to(device)

            input_map = {"rgb" : rgb_batch, "depth" : depth_batch}
            dt.measure_and_store(input_map)
            pbar.update(1)

        pbar.close()

        rgb_batch, depth_batch = next(iter(synth_loader)) #visualize one batch
        rgb_batch = rgb_batch.to(device)
        depth_batch = depth_batch.to(device)
        input_map = {"rgb": rgb_batch, "depth": depth_batch}
        if (global_config.plot_enabled == 1):
            dt.visualize_results(input_map, "FCity")
        dt.report_metrics("FCity")

        kitti_rgb_path = "X:/KITTI Depth Test/val_selection_cropped/image/*.png"
        kitti_depth_path = "X:/KITTI Depth Test/val_selection_cropped/groundtruth_depth/*.png"

        # compute total progress
        global_config.general_config["test_size"] = 64
        kitti_loader, dataset_count = dataset_loader.load_kitti_test_dataset(kitti_rgb_path, kitti_depth_path)
        dt = depth_tester.DepthTester(device)
        start_epoch = global_config.general_config["current_epoch"]
        print("---------------------------------------------------------------------------")
        print("Started kitti test loop for mode: depth", " Set start epoch: ", start_epoch)
        print("---------------------------------------------------------------------------")
        steps = general_config["test_size"]
        needed_progress = int(dataset_count / steps) + 1
        current_progress = 0
        pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
        pbar.update(current_progress)

        for i, (rgb_batch, depth_batch) in enumerate(kitti_loader, 0):
            rgb_batch = rgb_batch.to(device)
            depth_batch = depth_batch.to(device)

            input_map = {"rgb": rgb_batch, "depth": depth_batch}
            dt.measure_and_store(input_map)
            pbar.update(1)

        pbar.close()

        rgb_batch, depth_batch = next(iter(kitti_loader))  # visualize one batch
        rgb_batch = rgb_batch.to(device)
        depth_batch = depth_batch.to(device)
        input_map = {"rgb": rgb_batch, "depth": depth_batch}
        if (global_config.plot_enabled == 1):
            dt.visualize_results(input_map, "KITTI")
        dt.report_metrics("KITTI")



if __name__ == "__main__":
    main(sys.argv)