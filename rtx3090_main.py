import os
import multiprocessing
import time


def train_depth():
    os.system("python \"train_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.13\" "
              "--iteration=1")

    os.system("python \"train_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.13\" "
              "--iteration=2")

    # FOR TESTING
    # os.system("python \"train_main.py\" --server_config=3 --img_to_load=-1 "
    #           "--plot_enabled=0 --save_every_iter=200 --network_version=\"depth_v01.12\" "
    #           "--iteration=1")

def test_depth():
    os.system("python \"test_main.py\" --server_config=3 --img_to_load=1000 --plot_enabled=1 --network_version=\"depth_v01.13\" "
              "--iteration=1")
    #
    # os.system("python \"test_main.py\" --server_config=3 --img_to_load=-1 --plot_enabled=1 --network_version=\"depth_v01.12\" "
    #           "--iteration=4")

def train_img2img():
    os.system("python \"train_img2img_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=200 --network_version=\"synth2real_v01.00\" "
              "--iteration=1")

    # os.system("python \"train_img2img_main.py\" --server_config=3 --img_to_load=1000 "
    #           "--plot_enabled=0 --save_every_iter=200 --network_version=\"synth2real_v01.06\" "
    #           "--iteration=1")

def test_img2img():
    os.system("python \"test_img2img_main.py\" --server_config=3 --img_to_load=1000 "
              "--plot_enabled=1 --network_version=\"synth2real_v01.01\" "
              "--iteration=3")

    os.system("python \"test_img2img_main.py\" --server_config=3 --img_to_load=1000 "
              "--plot_enabled=1 --network_version=\"synth2real_v01.01\" "
              "--iteration=4")

def main():
    train_depth()
    # test_depth()
    #
    # train_img2img()
    #  test_img2img()
    # os.system("shutdown /s /t 1")

if __name__ == "__main__":
    main()