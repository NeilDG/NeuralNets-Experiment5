import multiprocessing
import os
import time

import GPUtil

def train_proper(gpu_device):
    os.system("python \"train_main.py\" --server_config=1 --img_to_load=-1 --cuda_device=" +gpu_device+ " "
             "--plot_enabled=0 --network_version=\"depth_v01.02\" "
             "--iteration=1")

    os.system("python \"train_main.py\" --server_config=1 --img_to_load=-1 --cuda_device=" +gpu_device+ " "
             "--plot_enabled=0 --network_version=\"depth_v01.02\" "
             "--iteration=2")

    os.system("python \"train_main.py\" --server_config=1 --img_to_load=-1 --cuda_device=" +gpu_device+ " "
             "--plot_enabled=0 --network_version=\"depth_v01.02\" "
             "--iteration=3")

    os.system("python \"train_main.py\" --server_config=1 --img_to_load=-1 --cuda_device=" +gpu_device+ " "
             "--plot_enabled=0 --network_version=\"depth_v01.02\" "
             "--iteration=4")

    os.system("python \"train_main.py\" --server_config=1 --img_to_load=-1 --cuda_device=" +gpu_device+ " "
             "--plot_enabled=0 --network_version=\"depth_v01.02\" "
             "--iteration=5")

    os.system("python \"train_main.py\" --server_config=1 --img_to_load=-1 --cuda_device=" +gpu_device+ " "
             "--plot_enabled=0 --network_version=\"depth_v01.02\" "
             "--iteration=6")

    os.system("python \"train_main.py\" --server_config=1 --img_to_load=-1 --cuda_device=" +gpu_device+ " "
             "--plot_enabled=0 --network_version=\"depth_v01.02\" "
             "--iteration=7")

    os.system("python \"train_main.py\" --server_config=1 --img_to_load=-1 --cuda_device=" +gpu_device+ " "
             "--plot_enabled=0 --network_version=\"depth_v01.02\" "
             "--iteration=10")

    os.system("python \"train_main.py\" --server_config=1 --img_to_load=-1 --cuda_device=" +gpu_device+ " "
             "--plot_enabled=0 --network_version=\"depth_v01.02\" "
             "--iteration=11")
def main():
    EXECUTION_TIME_IN_HOURS = 12
    execution_seconds = 3600 * EXECUTION_TIME_IN_HOURS

    GPUtil.showUtilization()
    device_id = GPUtil.getFirstAvailable(maxMemory=0.1, maxLoad=0.1, attempts=2500, interval=30, verbose=True)
    gpu_device = "cuda:" + str(device_id[0])
    print("Available GPU device found: ", gpu_device)

    p = multiprocessing.Process(target=train_proper, name="train_proper", args=(gpu_device,))
    p.start()

    time.sleep(execution_seconds) #causes p to execute code for X seconds. 3600 = 1 hour

    #terminate
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    print("\n Process " +p.name+ " has finished execution.")
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    p.terminate()
    p.join()


if __name__ == "__main__":
    main()