import os.path
import torch
import os

from config.network_config import ConfigHolder
from utils import tensor_utils

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.transforms.functional
import global_config
import kornia
from pathlib import Path
import kornia.augmentation as K

class DepthDataset(data.Dataset):
    def __init__(self, img_length, rgb_list, exr_list, transform_config):
        config_holder = ConfigHolder.getInstance()
        self.augment_mode = config_holder.get_network_attribute("augment_key", "none")
        self.use_tanh = config_holder.get_network_attribute("use_tanh", False)
        self.img_length = img_length
        self.rgb_list = rgb_list
        self.exr_list = exr_list
        self.transform_config = transform_config

        self.norm_op = transforms.Normalize((0.5, ), (0.5, ))

        if ("augmix" in self.augment_mode and self.transform_config == 1):
            self.initial_op = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256), antialias=True),
                transforms.AugMix(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor()])
        else:
            self.initial_op = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256), antialias=True),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor()
            ])

        if (self.transform_config == 1):
            patch_size = config_holder.get_network_attribute("patch_size", 32)
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = (256, 256)

    def __getitem__(self, idx):
        try:
            state = torch.get_rng_state()
            rgb_img = cv2.imread(self.rgb_list[idx])
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = self.initial_op(rgb_img)

            torch.set_rng_state(state)
            # depth_img = cv2.imread(self.exr_list[idx], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth_img = cv2.imread(self.exr_list[idx])
            depth_img = depth_img.astype(np.uint8)
            depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
            depth_img = 1.0 - self.initial_op(depth_img)

            if (self.transform_config == 1):
                crop_indices = transforms.RandomCrop.get_params(rgb_img, output_size=self.patch_size)
                i, j, h, w = crop_indices

                rgb_img = transforms.functional.crop(rgb_img, i, j, h, w)
                depth_img = transforms.functional.crop(depth_img, i, j, h, w)

            if(self.use_tanh):
                rgb_img = self.norm_op(rgb_img)
                depth_img = self.norm_op(depth_img)

        except Exception as e:
            print("Failed to load: ", self.rgb_list[idx], self.exr_list[idx])
            print("ERROR: ", e)

            rgb_img = None
            depth_img = None

        return rgb_img, depth_img

    def __len__(self):
        return self.img_length

class KittiDepthDataset(data.Dataset):
    def __init__(self, img_length, rgb_list, depth_list):
        self.img_length = img_length
        self.rgb_list = rgb_list
        self.depth_list = depth_list

        # self.norm_op = transforms.Normalize((0.5, ), (0.5, ))

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((88, 304), antialias=True), #divide by 4 KITTI size
            transforms.ToTensor()
        ])

        self.depth_op = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((88, 304), antialias=True),  # divide by 4 KITTI size
        ])

    def __getitem__(self, idx):
        rgb_img = cv2.imread(self.rgb_list[idx])
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = self.initial_op(rgb_img)

        # depth_img = tensor_utils.kitti_depth_read(self.depth_list[idx])
        depth_img = cv2.imread(self.depth_list[idx])
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
        depth_img = self.depth_op(depth_img)

        # rgb_img = self.norm_op(rgb_img)
        # depth_img = self.norm_op(depth_img)

        return rgb_img, depth_img

    def __len__(self):
        return self.img_length

class PairedImageDataset(data.Dataset):
    def __init__(self, a_list, b_list, transform_config):
        self.a_list = a_list
        self.b_list = b_list
        self.transform_config = transform_config

        config_holder = ConfigHolder.getInstance()
        self.augment_mode = config_holder.get_network_attribute("augment_key", "none")
        self.use_tanh = config_holder.get_network_attribute("use_tanh", False)

        if (self.transform_config == 1):
            patch_size = config_holder.get_network_attribute("patch_size", 32)
        else:
            patch_size = 256

        self.patch_size = (patch_size, patch_size)
        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor()
        ])

        self.norm_op = transforms.Normalize((0.5, ), (0.5, ))

    def __getitem__(self, idx):
        a_img = cv2.imread(self.a_list[idx])
        a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
        a_img = self.initial_op(a_img)

        b_img = cv2.imread(self.b_list[(idx % len(self.b_list))])
        b_img = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
        b_img = self.initial_op(b_img)

        if(self.use_tanh):
            a_img = self.norm_op(a_img)
            b_img = self.norm_op(b_img)

        return a_img, b_img

    def __len__(self):
        return len(self.a_list)
