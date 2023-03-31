import global_config
from abc import abstractmethod

from config.network_config import ConfigHolder
from model import ffa_gan, unet_gan, usi3d_gan
from model import vanilla_cycle_gan as cycle_gan
from model import monodepth_gan
from model import involution_gan
import torch
import torch.nn as nn


class NetworkCreator():
    def __init__(self, gpu_device):
        self.gpu_device = gpu_device

    def constant_weight_init(self, m, value):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, value)

        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def initialize_depth_network(self):
        network_config = ConfigHolder.getInstance().get_network_config()
        model_type = network_config["model_type"]

        D_A = cycle_gan.Discriminator(input_nc=1).to(self.gpu_device)  # use CycleGAN's discriminator
        if (model_type == 1):
            G_A = cycle_gan.Generator(input_nc=network_config["input_nc"], output_nc=1, n_residual_blocks=network_config["num_blocks"],
                                      dropout_rate=network_config["dropout_rate"], use_cbam=network_config["use_cbam"]).to(self.gpu_device)
            # self.constant_weight_init(G_A, 0.25)
        elif (model_type == 2):
            G_A = unet_gan.UnetGenerator(input_nc=network_config["input_nc"], output_nc=1, num_downs=network_config["num_blocks"]).to(self.gpu_device)
        elif (model_type == 3):
            G_A = ffa_gan.FFAGrey(network_config["num_blocks"], dropout_rate=network_config["dropout_rate"]).to(self.gpu_device)
        elif(model_type == 4):
            G_A = cycle_gan.GeneratorInvolution(input_nc=network_config["input_nc"], output_nc=1, n_residual_blocks=network_config["num_blocks"],
                                      dropout_rate=network_config["dropout_rate"]).to(self.gpu_device)
        elif(model_type == 5):
            G_A = monodepth_gan.I2D().to(self.gpu_device)
            self.constant_weight_init(G_A, 0.5)
        elif(model_type == 6):
            G_A = cycle_gan.SpectralGenerator(input_nc=network_config["input_nc"], output_nc=1, n_residual_blocks=network_config["num_blocks"],
                                              dropout_rate=network_config["dropout_rate"]).to(self.gpu_device)

            D_A = cycle_gan.SpectralDiscriminator(input_nc=1).to(self.gpu_device)
        else:
            G_A = ffa_gan.FFADepth(network_config["num_blocks"], dropout_rate=network_config["dropout_rate"]).to(self.gpu_device)


        return G_A, D_A

    def initialize_img2img_network(self):
        network_config = ConfigHolder.getInstance().get_network_config()
        model_type = network_config["model_type"]

        D_A = cycle_gan.Discriminator(input_nc=3).to(self.gpu_device)  # use CycleGAN's discriminator

        if (model_type == 1):
            G_A = cycle_gan.Generator(input_nc=network_config["input_nc"], output_nc=3, n_residual_blocks=network_config["num_blocks"],
                                      dropout_rate=network_config["dropout_rate"], use_cbam=network_config["use_cbam"], norm=network_config["norm_mode"]).to(self.gpu_device)
        elif (model_type == 2):
            G_A = unet_gan.UnetGenerator(input_nc=network_config["input_nc"], output_nc=3, num_downs=network_config["num_blocks"]).to(self.gpu_device)
        elif (model_type == 3):
            print("Using AdainGEN")
            params = {'dim': 64,  # number of filters in the bottommost layer
                      'mlp_dim': 256,  # number of filters in MLP
                      'style_dim': 8,  # length of style code
                      'n_layer': 3,  # number of layers in feature merger/splitor
                      'activ': 'relu',  # activation function [relu/lrelu/prelu/selu/tanh]
                      'n_downsample': 2,  # number of downsampling layers in content encoder
                      'n_res': network_config["num_blocks"],  # number of residual blocks in content encoder/decoder
                      'pad_type': 'reflect'}
            G_A = usi3d_gan.AdaINGen(input_dim=3, output_dim=3, params=params).to(self.gpu_device)
        else:
            G_A = cycle_gan.SpectralGenerator(input_nc=network_config["input_nc"], output_nc=3, n_residual_blocks=network_config["num_blocks"],
                                              dropout_rate=network_config["dropout_rate"], norm=network_config["norm_mode"]).to(self.gpu_device)

            D_A = cycle_gan.SpectralDiscriminator(input_nc=3).to(self.gpu_device)

        return G_A, D_A


class AbstractIIDTrainer():
    def __init__(self, gpu_device):
        self.gpu_device = gpu_device

        network_config = ConfigHolder.getInstance().get_network_config()
        self.g_lr = network_config["g_lr"]
        self.d_lr = network_config["d_lr"]

    @abstractmethod
    def initialize_train_config(self):
        pass

    @abstractmethod
    def initialize_dict(self):
        # what to store in visdom?
        pass

    @abstractmethod
    #follows a hashmap style lookup
    def train(self, epoch, iteration, input_map, target_map):
        pass

    @abstractmethod
    def is_stop_condition_met(self):
        pass

    @abstractmethod
    def test(self, input_map):
        pass

    @abstractmethod
    def visdom_plot(self, iteration):
        pass

    @abstractmethod
    def visdom_visualize(self, input_map, label="Train"):
        pass

    @abstractmethod
    def visdom_infer(self, input_map):
        pass

    @abstractmethod
    def load_saved_state(self):
        pass

    @abstractmethod
    def save_states(self, epoch, iteration, is_temp:bool):
        pass