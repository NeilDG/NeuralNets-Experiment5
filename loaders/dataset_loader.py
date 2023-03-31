import glob
import random
import torch

import global_config
from config.network_config import ConfigHolder
from loaders import image_datasets
from torch.utils import data

def load_train_dataset(rgb_path, exr_path):
    network_config = ConfigHolder.getInstance().get_network_config()
    general_config = global_config.general_config
    exr_list = glob.glob(exr_path)
    rgb_list = glob.glob(rgb_path)

    for i in range(0, network_config["dataset_repeats"]): #TEMP: formerly 0-1
        rgb_list += rgb_list
        exr_list += exr_list

    print("Length of images: %d %d" % (len(rgb_list), len(exr_list)))

    temp_list = list(zip(rgb_list, exr_list))
    random.shuffle(temp_list)

    rgb_list, exr_list = zip(*temp_list)
    img_length = len(rgb_list)
    print("Length of images: %d %d"  % (img_length, len(exr_list)))

    data_loader = torch.utils.data.DataLoader(
        image_datasets.DepthDataset(img_length, rgb_list, exr_list, 1),
        batch_size=global_config.load_size,
        num_workers=general_config["num_workers"],
        shuffle=False
    )

    return data_loader, len(rgb_list)

def load_test_dataset(rgb_path, exr_path):
    general_config = global_config.general_config

    exr_list = glob.glob(exr_path)
    rgb_list = glob.glob(rgb_path)

    temp_list = list(zip(rgb_list, exr_list))
    random.shuffle(temp_list)

    rgb_list, exr_list = zip(*temp_list)
    img_length = len(rgb_list)
    print("Length of images: %d %d"  % (img_length, len(exr_list)))

    data_loader = torch.utils.data.DataLoader(
        image_datasets.DepthDataset(img_length, rgb_list, exr_list, 2),
        batch_size=general_config["test_size"],
        num_workers=2,
        shuffle=False
    )

    return data_loader, len(rgb_list)

def load_train_img2img_dataset(a_path, b_path):
    network_config = ConfigHolder.getInstance().get_network_config()
    general_config = global_config.general_config
    a_list = glob.glob(a_path)
    b_list = glob.glob(b_path)
    a_list_dup = glob.glob(a_path)
    b_list_dup = glob.glob(b_path)

    if (global_config.img_to_load > 0):
        a_list = a_list[0: global_config.img_to_load]
        b_list = b_list[0: global_config.img_to_load]
        a_list_dup = a_list_dup[0: global_config.img_to_load]
        b_list_dup = b_list_dup[0: global_config.img_to_load]

    for i in range(0, network_config["dataset_a_repeats"]): #TEMP: formerly 0-1
        a_list += a_list_dup

    for i in range(0, network_config["dataset_b_repeats"]): #TEMP: formerly 0-1
        b_list += b_list_dup

    random.shuffle(a_list)
    random.shuffle(b_list)

    img_length = len(a_list)
    print("Length of images: %d %d"  % (img_length, len(b_list)))

    num_workers = general_config["num_workers"]
    data_loader = torch.utils.data.DataLoader(
        image_datasets.PairedImageDataset(a_list, b_list, 1),
        batch_size=global_config.load_size,
        num_workers=num_workers
    )

    return data_loader, img_length

def load_test_img2img_dataset(a_path, b_path):
    a_list = glob.glob(a_path)
    b_list = glob.glob(b_path)

    if (global_config.img_to_load > 0):
        a_list = a_list[0: global_config.img_to_load]
        b_list = b_list[0: global_config.img_to_load]

    random.shuffle(a_list)
    random.shuffle(b_list)

    img_length = len(a_list)
    print("Length of images: %d %d" % (img_length, len(b_list)))

    data_loader = torch.utils.data.DataLoader(
        image_datasets.PairedImageDataset(a_list, b_list, 1),
        batch_size=global_config.general_config["test_size"],
        num_workers=1
    )

    return data_loader, img_length

def load_kitti_test_dataset(rgb_path, depth_path):
    general_config = global_config.general_config

    rgb_list = glob.glob(rgb_path)
    depth_list = glob.glob(depth_path)

    temp_list = list(zip(rgb_list, depth_list))
    random.shuffle(temp_list)

    rgb_list, depth_list = zip(*temp_list)
    img_length = len(rgb_list)
    print("Length of images: %d %d" % (img_length, len(depth_list)))

    data_loader = torch.utils.data.DataLoader(
        image_datasets.KittiDepthDataset(img_length, rgb_list, depth_list),
        batch_size=general_config["test_size"],
        num_workers=2,
        shuffle=False
    )

    return data_loader, len(rgb_list)

