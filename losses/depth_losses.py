import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import global_config


#
# Commonly used losses for depth predictions
# Taken from: https://github.com/haofengac/MonoDepth-FPN-PyTorch/blob/master/main_fpn.py

def imgrad(img):
    device = global_config.general_config["cuda_device"]
    img = torch.mean(img, 1, True)
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False).to(device)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.to(device)
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False).to(device)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.to(device)
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

    #     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))

    return grad_y, grad_x

def imgrad_yx(img):
    N, C, _, _ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N, C, -1), grad_x.view(N, C, -1)), dim=1)

class RMSEDepthLoss(nn.Module):
    def __init__(self):
        super(RMSEDepthLoss, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode='bilinear')
        loss = torch.sqrt(torch.mean(torch.abs(10. * real - 10. * fake) ** 2))
        return loss


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

    #L1 norm
    def forward(self, fake, real):
        grad_fake = imgrad_yx(fake)
        grad_real = imgrad_yx(real)

        return torch.sum(torch.mean(torch.abs(grad_real - grad_fake)))

# From monodepth2 by godard
class DepthSmoothnessLoss(nn.Module):
    def __init__(self):
        super(DepthSmoothnessLoss, self).__init__()

    def forward(self, pred, target):
        """Computes the smoothness loss for a disparity image
            The color image is used for edge-aware smoothness
            """
        grad_disp_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        grad_disp_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

