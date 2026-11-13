#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --nodelist=node-03
#SBATCH --cpus-per-task=8

source ~/.bashrc
conda activate job_new
cd ..

# args are forwarded, e.g.: sbatch scripts/classify_frames.sh -- --config 2025 --fold 0
# (all 3 configs x 5 folds: sbatch scripts/train_classify_xdomain.sh)
python -u -m informative_frames.classify_frames "$@"
