#!/bin/bash
#SBATCH -J ZDB
#SBATCH --partition=debug
#SBATCH --qos=240c-1h_debug
#SBATCH --ntasks=1
#SBATCH --output=script_debug.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

NETWORK_VERSION=$1
ITERATION=$2
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
echo "Set network to "$NETWORK_VERSION " Set iteration to "$ITERATION
nvidia-smi

# prepare working environment
module load anaconda/3-2021.11
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python train_main.py \
--network_version=$NETWORK_VERSION --iteration=$ITERATION --server_config=0 --plot_enabled=0 --img_to_load=-1

conda deactivate

