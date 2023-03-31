import itertools
import sys
from optparse import OptionParser
import random
import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
from config.network_config import ConfigHolder
from loaders import dataset_loader
import global_config
from utils import plot_utils
from testers import img2img_tester
from tqdm import tqdm
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
    global_config.img_to_load = opts.img_to_load
    global_config.general_config["cuda_device"] = opts.cuda_device
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
        global_config.path = "C:/Datasets/SynthV3_Raw/{dataset_version}/sequence.0/"
        print("Using RTX 2080Ti configuration. Workers: ", global_config.general_config["num_workers"])

    elif (global_config.server_config == 3):  # RTX 3090 PC
        global_config.general_config["num_workers"] = 12
        global_config.a_path = "X:/Places Dataset/*.jpg"
        global_config.b_path = "X:/SynthV3_Raw/{dataset_version}/sequence.0/*.camera.png"
        global_config.batch_size = network_config["batch_size"][0]
        global_config.load_size = network_config["load_size"][0]
        print("Using RTX 3090 configuration. Workers: ", global_config.general_config["num_workers"])

    elif (global_config.server_config == 4):  # RTX 2070 PC @RL208
        global_config.general_config["num_workers"] = 4
        global_config.path = "D:/Datasets/SynthV3_Raw/{dataset_version}/sequence.0/"
        print("Using RTX 2070 @RL208 configuration. Workers: ", global_config.general_config["num_workers"])

    elif (global_config.server_config == 5):  # RTX 3060 PC Titan
        global_config.general_config["num_workers"] = 12
        global_config.path = "X:/SynthV3_Raw/{dataset_version}/sequence.0/"
        print("Using TITAN RTX 3060 configuration. Workers: ", global_config.general_config["num_workers"])

    elif (global_config.server_config == 6):  # RTX 2080Ti @TITAN
        global_config.general_config["num_workers"] = 12
        global_config.a_path = "/home/neildelgallego/SynthV3_Raw/{dataset_version}/sequence.0/"
        print("Using TITAN RTX 2080Ti configuration. Workers: ", global_config.general_config["num_workers"])

    global_config.b_path = global_config.b_path.format(dataset_version=network_config["dataset_version"])


def main(argv):
    (opts, args) = parser.parse_args(argv)
    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    manualSeed = 0
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    yaml_config = "./hyperparam_tables/{network_version}.yaml"
    yaml_config = yaml_config.format(network_version=opts.network_version)
    hyperparam_path = "./hyperparam_tables/synth2real_iter.yaml"
    with open(yaml_config) as f, open(hyperparam_path) as h:
        ConfigHolder.initialize(yaml.load(f, SafeLoader), yaml.load(h, SafeLoader))

    update_config(opts)
    print(opts)
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (global_config.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    network_config = ConfigHolder.getInstance().get_network_config()
    hyperparam_config = ConfigHolder.getInstance().get_hyper_params()
    network_iteration = global_config.general_config["iteration"]
    hyperparams_table = hyperparam_config["hyperparams"][network_iteration]
    print("Network iteration:", str(network_iteration), ". Hyper parameters: ", hyperparams_table, " Learning rates: ", network_config["g_lr"], network_config["d_lr"])

    a_path = global_config.a_path
    b_path = global_config.b_path

    print("Dataset path A: ", a_path)
    print("Dataset path B: ", b_path)

    plot_utils.VisdomReporter.initialize()

    test_loader_a, test_count = dataset_loader.load_test_img2img_dataset(a_path, b_path)

    img2img_t = img2img_tester.Img2ImgTester(device)
    general_config = global_config.general_config
    start_epoch = general_config["current_epoch"]
    print("---------------------------------------------------------------------------")
    print("Started synth test loop for mode: synth2real", " Set start epoch: ", start_epoch)
    print("---------------------------------------------------------------------------")

    # compute total progress
    general_config["test_size"] = 256
    steps = general_config["test_size"]
    needed_progress = int(test_count / steps) + 1
    current_progress = 0
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)

    with torch.no_grad():
        for i, (a_batch, b_batch) in enumerate(test_loader_a, 0):
            a_batch = a_batch.to(device)
            b_batch = b_batch.to(device)

            input_map = {"img_a" : a_batch, "img_b" : b_batch}
            img2img_t.measure_and_store(input_map)
            pbar.update(1)

        pbar.close()

        a_test_batch, b_test_batch = next(iter(test_loader_a))
        a_test_batch = a_test_batch.to(device)
        b_test_batch = b_test_batch.to(device)
        input_map = {"img_a": a_test_batch, "img_b": b_test_batch}
        if (global_config.plot_enabled == 1):
            img2img_t.visualize_results(input_map, "Synth2Real")
        img2img_t.report_metrics("Synth2Real")

if __name__ == "__main__":
    main(sys.argv)