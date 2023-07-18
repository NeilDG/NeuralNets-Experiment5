import torch.nn as nn
import torch.nn.functional as F
from config.network_config import ConfigHolder
from model.modules import cbam_module, common_blocks
from ITTR_pytorch import HPB

def xavier_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data, nn.init.calculate_gain('tanh'))
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def clamp(value, max):
    if(value > max):
        return max
    else:
        return value


class ITTRTransformer(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, downsampling_blocks = 2, perception_blocks=9, dropout_rate = 0.0, norm = "batch"):
        super(ITTRTransformer, self).__init__()

        print("Set CycleGAN norm to: ", norm, "Dropout rate: ", dropout_rate)

        # Initial convolution block
        model = [   nn.ReflectionPad2d(2),
                    nn.Conv2d(input_nc, 64, 7),
                    common_blocks.NormBlock(64, norm),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(downsampling_blocks):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        common_blocks.NormBlock(out_features, norm),
                        nn.ReLU(inplace=True)
                    ]

            model +=[nn.Dropout2d(p = dropout_rate)]
            in_features = out_features
            out_features = clamp(in_features*2, 32768)

        # Residual blocks
        for _ in range(perception_blocks):
            model += [HPB(dim = in_features, ff_dropout=dropout_rate, attn_dropout=dropout_rate, dim_head=16)]

        # Upsampling
        out_features = in_features//2
        for _ in range(downsampling_blocks):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        common_blocks.NormBlock(out_features, norm),
                        nn.ReLU(inplace=True)]

            model += [nn.Dropout2d(p=dropout_rate)]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    common_blocks.LastLayerBlock() ]

        self.model = nn.Sequential(*model)
        self.model.apply(xavier_weights_init)

    def forward(self, x):
        return self.model(x)