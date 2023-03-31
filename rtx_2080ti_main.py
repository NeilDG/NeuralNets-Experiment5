import os

def train_depth():
    os.system("python \"train_main.py\" --server_config=2 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.14\" "
              "--iteration=5")

    os.system("python \"train_main.py\" --server_config=2 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.14\" "
              "--iteration=6")

    os.system("python \"train_main.py\" --server_config=2 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.14\" "
              "--iteration=7")

def test_depth():
    os.system("python \"test_main.py\" --server_config=2 --img_to_load=-1 --plot_enabled=1 --network_version=\"depth_v01.04\" "
              "--iteration=12")

def train_img2img():
    os.system("python \"train_img2img_main.py\" --server_config=2 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=200 --network_version=\"synth2real_v01.00\" "
              "--iteration=3")

def test_img2img():
    os.system("python \"test_img2img_main.py\" --server_config=2 --img_to_load=1000 "
              "--plot_enabled=1 --network_version=\"synth2real_v01.00\" "
              "--iteration=1")

def main():
    train_depth()
    # test_depth()
    # train_img2img()
    os.system("shutdown /s /t 1")

if __name__ == "__main__":
    main()