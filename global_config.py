# -*- coding: utf-8 -*-
import os

#========================================================================#
OPTIMIZER_KEY = "optimizer"
GENERATOR_KEY = "generator"
DISCRIMINATOR_KEY = "discriminator"

LAST_METRIC_KEY = "last_metric"

plot_enabled = 1
disable_progress_bar = False
save_every_iter = 500

#Running on local = 0, Running on COARE = 1, Running on CCS server = 2
server_config = 0
general_config = {
    "num_workers" : 12,
    "cuda_device" : "cuda:0",
    "network_version" : "VXX.XX",
    "iteration" : 1,
    "current_epoch" : 0,
    "test_size" : 16,
}

batch_size = -1
load_size = -1
img_to_load = -1

path = ""
exr_path = ""
rgb_path = ""

a_path = ""
b_path = ""