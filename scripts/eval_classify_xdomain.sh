#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Evaluate the cross-dataset informative-frame classifier experiment: every trained
# classifier (informative_frames/models_xdomain/cls_<config>_fold_<i>.pth, config in
# 2020/2025/both) is evaluated on both datasets, using the frames cropped by the same
# config (frames_cropped_paxos<eval_ds>_by_<config>). Same rule as
# cropping/evaluate_segmentation.py: in-dataset -> fold-i model on the fold-i val split
# (5 points per bar); cross-dataset -> 5-model mean-sigmoid-probability ensemble on ALL
# frames of the other dataset (one hatched bar). Writes per_image_metrics.csv,
# summary.csv, bar_summary.csv and classification_eval.png (3 subplots: ROC-AUC, PR-AUC,
# F1; 6 bars each) to informative_frames/eval_xdomain/.
#
# Prerequisite: sbatch scripts/train_classify_xdomain.sh
#
# Usage: sbatch scripts/eval_classify_xdomain.sh

source ~/.bashrc
conda activate job_new

python -u -m informative_frames.evaluate_classification \
  --models_dir informative_frames/models_xdomain \
  --img_base data/informative_frames \
  --split_dir informative_frames \
  --out_dir informative_frames/eval_xdomain
