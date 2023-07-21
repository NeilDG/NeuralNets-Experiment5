#!/bin/bash
#SBATCH -J DOWNLOAD
#SBATCH --partition=serial
#SBATCH --qos=84c-1d_serial
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --output=script_download_2.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

#About this script:
#Download of dataset
SERVER_CONFIG=$1

module load anaconda/3-2021.11
module load cuda/10.1_cudnn-7.6.5
source activate NeilGAN_V2

#do fresh install
#pip-review --local --auto
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip install scikit-learn
#pip install scikit-image
#pip install visdom
#pip install kornia
#pip install opencv-python==4.5.5.62
#pip install --upgrade pillow
#pip install gputil
#pip install matplotlib
#pip install --upgrade --no-cache-dir gdown
#pip install PyYAML

if [ $SERVER_CONFIG == 0 ]
then
  srun python "gdown_download.py" --server_config=$SERVER_CONFIG
elif [ $SERVER_CONFIG == 5 ]
then
  python3 "gdown_download.py" --server_config=$SERVER_CONFIG
else
  python "gdown_download.py" --server_config=$SERVER_CONFIG
fi


if [ $SERVER_CONFIG == 0 ]
then
  OUTPUT_DIR="/scratch3/neil.delgallego/SynthV3_Raw/"
elif [ $SERVER_CONFIG == 4 ]
then
  OUTPUT_DIR="D:/NeilDG/Datasets/SynthV3_Raw/"
elif [ $SERVER_CONFIG == 5 ]
then
  OUTPUT_DIR="/home/neildelgallego/SynthV3_Raw/"
else
  OUTPUT_DIR="/home/jupyter-neil.delgallego/SynthV3_Raw/"
fi

#DATASET_NAME="KITTI Depth Test"
#echo "$OUTPUT_DIR/$DATASET_NAME.zip"
#unzip "$OUTPUT_DIR/$DATASET_NAME.zip" -d "$OUTPUT_DIR"

#DATASET_NAME="v05_iid"
#echo "$OUTPUT_DIR/$DATASET_NAME.zip"
#unzip "$OUTPUT_DIR/$DATASET_NAME.zip" -d "$OUTPUT_DIR"

#DATASET_NAME="v06_iid_base/v06_iid"
#zip -F "$OUTPUT_DIR/$DATASET_NAME.zip" --out "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"
#unzip "$OUTPUT_DIR/$DATASET_NAME+fixed.zip" -d "$OUTPUT_DIR"
#
#DATASET_NAME="v07_iid_base/v07_iid"
#zip -F "$OUTPUT_DIR/$DATASET_NAME.zip" --out "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"
#unzip "$OUTPUT_DIR/$DATASET_NAME+fixed.zip" -d "$OUTPUT_DIR"

DATASET_NAME="v09_iid_base/v09_iid"
zip -F "$OUTPUT_DIR/$DATASET_NAME.zip" --out "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"
unzip "$OUTPUT_DIR/$DATASET_NAME+fixed.zip" -d "$OUTPUT_DIR"

#DATASET_NAME="places_dataset_base/places"
#zip -F "$OUTPUT_DIR/$DATASET_NAME.zip" --out "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"
#unzip "$OUTPUT_DIR/$DATASET_NAME+fixed.zip" -d "$OUTPUT_DIR"

#
#zip -F "$OUTPUT_DIR/$DATASET_NAME.zip" --out "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"
#unzip "$OUTPUT_DIR/$DATASET_NAME+fixed.zip" -d "$OUTPUT_DIR"
#

if [ $SERVER_CONFIG == 5 ]
then
  python3 "titan2_main.py"
fi