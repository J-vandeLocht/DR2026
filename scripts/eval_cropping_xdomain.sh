#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Evaluate the cross-dataset cropping experiment: every trained model
# (cropping/models_xdomain/seg_<config>_fold_<i>.pth, config in 2020/2025/both)
# is evaluated on the fold-i val split of both datasets. Writes per-image metrics,
# summary CSVs and segmentation_eval.png (6 bars x 5 fold points per metric)
# to cropping/eval_xdomain/.
#
# Usage: sbatch scripts/eval_cropping_xdomain.sh

source ~/.bashrc
conda activate job_new

python -u -m cropping.evaluate_segmentation \
  --models_dir cropping/models_xdomain \
  --ann_dir cropping \
  --img_dir cropping/frames_to_annotate_larger \
  --out_dir cropping/eval_xdomain
