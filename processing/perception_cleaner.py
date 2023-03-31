import glob

import torch

from utils import plot_utils
from loaders import dataset_loader

def organize_dataset():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    plot_utils.VisdomReporter.initialize()
    visdom_reporter = plot_utils.VisdomReporter.getInstance()

    path = "X:/SynthV3_Raw/sequence.0/"
    exr_path = path + "*.exr"
    rgb_path = path + "*.camera.png"
    segmentation_path = path + "*.semantic segmentation.png"

    opts = {}
    opts["num_workers"] = 12
    opts["load_size"] = 64
    opts["cuda_device"] = "cuda:0"
    opts["train_config"] = 1

    test_loader = dataset_loader.load_custom_dataset(rgb_path, exr_path, segmentation_path, opts)
    for i, (rgb_batch, depth_batch, segmentation_batch) in enumerate(test_loader, 0):
        rgb_batch = rgb_batch.to(device)
        depth_batch = depth_batch.to(device)
        segmentation_batch = segmentation_batch.to(device)

        visdom_reporter.plot_image(rgb_batch, "Train RGB")
        visdom_reporter.plot_image(depth_batch, "Train Depth")
        visdom_reporter.plot_image(segmentation_batch, "Train Segmentation")

        break

    opts["train_config"] = 2
    test_loader = dataset_loader.load_custom_dataset(rgb_path, exr_path, segmentation_path, opts)
    for i, (rgb_batch, depth_batch, segmentation_batch) in enumerate(test_loader, 0):
        rgb_batch = rgb_batch.to(device)
        depth_batch = depth_batch.to(device)
        segmentation_batch = segmentation_batch.to(device)

        visdom_reporter.plot_image(rgb_batch, "Test RGB")
        visdom_reporter.plot_image(depth_batch, "Test Depth")
        visdom_reporter.plot_image(segmentation_batch, "Test Segmentation")

        break


def main():
    organize_dataset()

if __name__ == "__main__":
    main()