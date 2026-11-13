#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A100short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# Cross-dataset informative-frame classifier experiment: train the EfficientNet-B4
# selection model for configs 2020 / 2025 / both, each over 5 folds. Variant <config>
# trains on the frames cropped by the same cropping config
# (data/informative_frames/frames_cropped_paxos<ds>_by_<config>), with the folds from
# informative_frames/split_info_5fold_paxos<ds>_from_cropping.csv (built by
# informative_frames.make_splits_from_cropping.py from the cropping splits).
# Single-dataset runs train for double the epochs of the 'both' runs (the two cohorts
# are about the same size): BASE_EPOCHS for 'both', 2*BASE_EPOCHS otherwise.
# Early stopping: PATIENCE epochs without val PR-AUC improvement before stopping.
# Models are saved to $OUT_DIR/cls_<config>_fold_<fold>.pth.
#
# Prerequisites (run from the repo root first):
#   python -u -m informative_frames.make_splits_from_cropping
#   sbatch scripts/crop_frames_xdomain.sh     # the 6 cropped folders
#
# Usage: sbatch scripts/train_classify_xdomain.sh
#        CONFIGS="2025" bash scripts/train_classify_xdomain.sh   (subset of configs)

source ~/.bashrc
conda activate job_new

CONFIGS=${CONFIGS:-"2020 2025 both"}
BASE_EPOCHS=${BASE_EPOCHS:-10}
FOLDS=${FOLDS:-"0 1 2 3 4"}
IMG_BASE=${IMG_BASE:-data/informative_frames}
SPLITS_DIR=${SPLITS_DIR:-informative_frames}
OUT_DIR=${OUT_DIR:-informative_frames/models_xdomain}
PATIENCE=${PATIENCE:-3}   # early stopping: epochs without val PR-AUC improvement before stopping
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-8}

for cfg in $CONFIGS; do
  if [ "$cfg" = "both" ]; then epochs=$BASE_EPOCHS; else epochs=$((2 * BASE_EPOCHS)); fi
  for fold in $FOLDS; do
    echo "=== train classifier config=$cfg fold=$fold epochs=$epochs ==="
    python -u -m informative_frames.classify_frames \
      --config "$cfg" --fold "$fold" --epochs "$epochs" --patience "$PATIENCE" \
      --img_base "$IMG_BASE" --split_dir "$SPLITS_DIR" --models_dir "$OUT_DIR" \
      --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS"
  done
done

echo "All classifier xdomain training runs finished. Models in $OUT_DIR"
