import torch
import torch.nn as nn
import torch.nn.functional as F
from config.network_config import ConfigHolder

class LastLayerBlock(nn.Module):
    def __init__(self):
        super(LastLayerBlock, self).__init__()
        if (ConfigHolder.getInstance().get_network_attribute("use_tanh", False)):
            self.last_layer = nn.Tanh()
            # print("Using tanh")
        else:
            self.last_layer = nn.Sigmoid()

    def forward(self, x):
        return self.last_layer(x)

class NormBlock(nn.Module):
    def __init__(self, in_features, norm):
        super(NormBlock, self).__init__()
        if (norm == "batch"):
            self.norm_mode = nn.BatchNorm2d(in_features)
        else:
            self.norm_mode = nn.InstanceNorm2d(in_features)

    def forward(self, x):
        return self.norm_mode(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, norm):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        NormBlock(in_features, norm),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        NormBlock(in_features, norm)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)