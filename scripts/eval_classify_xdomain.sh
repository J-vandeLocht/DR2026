#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Evaluate the cross-dataset informative-frame classifier experiment: every trained
# classifier (informative_frames/models_xdomain/cls_<config>_fold_<i>.pth) is evaluated
# on both datasets, using the frames cropped by the same config
# (frames_cropped_paxos<eval_ds>_by_<config>). Same rule as
# cropping/evaluate_segmentation.py: in-dataset -> fold-i model on the fold-i val split
# (5 points per bar); cross-dataset -> 5-model mean-sigmoid-probability ensemble on ALL
# frames of the other dataset (one hatched bar). Writes per_image_metrics.csv,
# summary.csv, bar_summary.csv and classification_eval.png (3 subplots: ROC-AUC, PR-AUC,
# F1; one bar per trained config x eval dataset) to informative_frames/eval_xdomain/.
# The evaluated configs are auto-detected from the checkpoints in $MODELS_DIR, so a
# config that was not trained (e.g. "both") is simply left out of the plot.
#
# Prerequisite: sbatch scripts/train_classify_xdomain.sh
#
# Usage: sbatch scripts/eval_classify_xdomain.sh
#        CONFIGS="2020 2025" sbatch scripts/eval_classify_xdomain.sh   (explicit configs)

source ~/.bashrc
conda activate job_new

MODELS_DIR=${MODELS_DIR:-informative_frames/models_xdomain}
CONFIGS=${CONFIGS:-}   # empty = auto-detect trained configs from $MODELS_DIR
CFG_ARGS=()
if [ -n "$CONFIGS" ]; then CFG_ARGS=(--configs $CONFIGS); fi

python -u -m informative_frames.evaluate_classification \
  --models_dir "$MODELS_DIR" \
  --img_base data/informative_frames \
  --split_dir informative_frames \
  --out_dir informative_frames/eval_xdomain \
  "${CFG_ARGS[@]}"
