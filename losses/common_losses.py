import kornia.losses
from losses import depth_losses
import torch.nn as nn
import torch
from config.network_config import ConfigHolder

#
# Class to contain common losses used for training networks
#
class LossRepository():
    def __init__(self, gpu_device, iteration):
        self.gpu_device = gpu_device
        self.iteration = iteration
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.depth_smooth_loss = depth_losses.DepthSmoothnessLoss()
        self.gradient_loss = depth_losses.GradLoss()
        self.rmse_loss = depth_losses.RMSEDepthLoss()
        self.ssim_loss = kornia.losses.SSIMLoss(5)

    def compute_adversarial_loss(self, pred, target):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "adv_weight")
        use_bce = config_holder.get_hyper_params_weight(self.iteration, "is_bce")

        if (weight > 0.0 and use_bce == 0):
            return self.mse_loss(pred, target) * weight
        elif (weight > 0.0 and use_bce == 1):
            return self.bce_loss(pred, target) * weight
        else:
            return torch.zeros_like(self.l1_loss(pred, target))

    def compute_l1_loss(self, pred, target):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "l1_weight")
        if (weight > 0.0):
            return self.l1_loss(pred, target) * weight
        else:
            return torch.zeros_like(self.l1_loss(pred, target))

    def compute_l1_log_loss(self, pred, target):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "l1_log_weight")
        if(weight > 0.0):
            return self.l1_loss(torch.log(pred), torch.log(target)) * weight
        else:
            return torch.zeros_like(self.l1_loss(pred, target))

    def compute_mse_loss(self, pred, target):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "mse_weight")
        if (weight > 0.0):
            return self.mse_loss(pred, target) * weight
        else:
            return torch.zeros_like(self.l1_loss(pred, target))

    def compute_rmse_loss(self, pred, target):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "rmse_weight")
        if (weight > 0.0):
            return self.rmse_loss(pred, target) * weight
        else:
            return torch.zeros_like(self.l1_loss(pred, target))

    def compute_rmse_log_loss(self, pred, target):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "rmse_log_weight")
        if (weight > 0.0):
            return self.rmse_loss(torch.log(pred), torch.log(target)) * weight
        else:
            return torch.zeros_like(self.l1_loss(pred, target))

    def compute_depth_smoothness_loss(self, pred, target):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "disp_weight")
        if (weight > 0.0):
            return self.depth_smooth_loss(pred, target) * weight
        else:
            return torch.zeros_like(self.l1_loss(pred, target))

    def compute_grad_loss(self, pred, target):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "grad_weight")
        if (weight > 0.0):
            return self.gradient_loss(pred, target) * weight
        else:
            return torch.zeros_like(self.l1_loss(pred, target))

    def compute_ssim_loss(self, pred, target):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "ssim_weight")
        if (weight > 0.0):
            return self.ssim_loss(pred, target) * weight
        else:
            return torch.zeros_like(self.l1_loss(pred, target))