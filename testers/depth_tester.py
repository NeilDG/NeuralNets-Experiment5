from pathlib import Path

import cv2
import kornia.metrics.psnr
import torchvision

from config import network_config
from config.network_config import ConfigHolder
from trainers import abstract_iid_trainer
import global_config
import torch
import torch.cuda.amp as amp
import itertools
from model.modules import image_pool
from utils import plot_utils, tensor_utils
import torch.nn as nn
import numpy as np
from trainers import depth_trainer
from testers import depth_metrics

class DepthTester():
    def __init__(self, gpu_device):
        self.gpu_device = gpu_device
        self.dt = depth_trainer.DepthTrainer(self.gpu_device)
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()

        self.l1_results = []
        self.mse_results = []
        self.rmse_results = []
        self.rmse_log_results = []
        self.psnr_results = []

    #measures the performance of a given batch and stores it
    def measure_and_store(self, input_map):
        use_tanh = ConfigHolder.getInstance().get_network_attribute("use_tanh", False)
        rgb2target = self.dt.test(input_map)
        target_depth = input_map["depth"]

        if(use_tanh):
            rgb2target = tensor_utils.normalize_to_01(rgb2target)
            target_depth = tensor_utils.normalize_to_01(target_depth)

        depth_mask = torch.isfinite(target_depth) & (target_depth > 0.0)

        rgb2target = torch.masked_select(rgb2target, depth_mask)
        target_depth = torch.masked_select(target_depth, depth_mask)

        psnr_result = kornia.metrics.psnr(rgb2target, target_depth, torch.max(target_depth).item())
        self.psnr_results.append(psnr_result.item())

        l1_result = self.l1_loss(rgb2target, target_depth).cpu()
        self.l1_results.append(l1_result)

        mse_result = self.mse_loss(rgb2target, target_depth).cpu()
        self.mse_results.append(mse_result)

        # num_zeros_target = np.shape(torch.flatten(target_depth))[0] - torch.count_nonzero(target_depth).item()
        # num_zeros_pred = np.shape(torch.flatten(rgb2target))[0] - torch.count_nonzero(rgb2target).item()
        # print("Has zeros? ", num_zeros_pred, num_zeros_target)

        rmse_result = depth_metrics.torch_rmse(rgb2target, target_depth).item()
        self.rmse_results.append(rmse_result)

        rmse_log_result = depth_metrics.torch_rmse_log(target_depth,rgb2target).item()
        # print("Rmse log result: ", rmse_log_result)
        self.rmse_log_results.append(rmse_log_result)

    def visualize_results(self, input_map, dataset_title):
        version_name = network_config.ConfigHolder.getInstance().get_version_name()
        self.dt.visdom_visualize(input_map, "Test - " + version_name + " " + dataset_title)

    def save_image_set(self, file_names, input_map, a_key, b_key, dataset_title):
        version_name = network_config.ConfigHolder.getInstance().get_version_name()
        SAVE_PATH = "./results/" + version_name + "/" + dataset_title + "/"
        try:
            path = Path(SAVE_PATH + "/input/")
            path.mkdir(parents=True)

            path = Path(SAVE_PATH + "/target/")
            path.mkdir(parents=True)

            path = Path(SAVE_PATH + "/target-like/")
            path.mkdir(parents=True)
        except OSError as error:
            pass
            # print(SAVE_PATH + " already exists. Skipping.", error)

        generated = self.dt.test(input_map)
        input = input_map[a_key]
        ground_truth = input_map[b_key]

        generated = (generated - 0.85) * 2.5
        # ground_truth = kornia.enhance.add_weighted(ground_truth, 0.75, kornia.enhance.equalize(ground_truth), 0.25, 0.0)

        for i in range(0, len(file_names)):
            img_save_file = SAVE_PATH + "/input/" + file_names[i] + ".png"
            torchvision.utils.save_image(input[i], img_save_file, normalize=True)

            img_save_file = SAVE_PATH + "/target/" + file_names[i] + ".png"
            torchvision.utils.save_image(ground_truth[i], img_save_file, normalize=False)
            ground_truth_img = cv2.imread(img_save_file)
            ground_truth_img = cv2.cvtColor(ground_truth_img, cv2.COLOR_BGR2GRAY)
            ground_truth_img = cv2.applyColorMap(ground_truth_img, cv2.COLORMAP_MAGMA)
            cv2.imwrite(img_save_file, ground_truth_img)


            img_save_file = SAVE_PATH + "/target-like/" + file_names[i] + ".png"
            torchvision.utils.save_image(generated[i], img_save_file, normalize=False)

        # print("Saved batch of images of size " + str(len(file_names)))

    def report_metrics(self, dataset_title):
        version_name = network_config.ConfigHolder.getInstance().get_version_name()

        psnr_mean = np.round(np.mean(self.psnr_results), 4)
        self.psnr_results.clear()

        l1_mean = np.round(np.float32(np.mean(self.l1_results)), 4) #ISSUE: ROUND to 4 sometimes cause inf
        self.l1_results.clear()

        mse_mean = np.round(np.mean(self.mse_results), 4)
        self.mse_results.clear()

        rmse_mean = np.round(np.mean(self.rmse_results), 4)
        self.rmse_results.clear()

        rmse_log_mean = np.round(np.mean(self.rmse_log_results), 4)
        self.rmse_log_results.clear()

        last_epoch = global_config.general_config["current_epoch"]
        self.visdom_reporter.plot_text(dataset_title + " Results - " + version_name + " Last epoch: " + str(last_epoch) + "<br>"
                                       + "PSNR: " +str(psnr_mean) + "<br>" 
                                       "Abs Rel: " + str(l1_mean) + "<br>"
                                        "Sqr Rel: " + str(mse_mean) + "<br>"
                                       "RMSE: " + str(rmse_mean) + "<br>"
                                       "RMSE log: " +str(rmse_log_mean) +"<br>")
