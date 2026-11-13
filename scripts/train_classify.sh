#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A100short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# Paxos2025 informative-frame classifier experiment: train the EfficientNet-B4
# selection model over the 5 folds, on the frames cropped by scripts/crop_frames.sh
# (data/informative_frames/frames_cropped), with the folds from
# informative_frames/split_info_5fold_paxos2025_from_cropping.csv (built by
# informative_frames.make_splits_from_cropping.py from the cropping splits).
# Early stopping: PATIENCE epochs without val PR-AUC improvement before stopping.
# Models are saved to $OUT_DIR/cls_fold_<fold>.pth.
#
# Prerequisites (run from the repo root first):
#   python -u -m informative_frames.make_splits_from_cropping
#   sbatch scripts/crop_frames.sh
#
# Usage: sbatch scripts/train_classify.sh
#        FOLDS="0" bash scripts/train_classify.sh   (subset of folds)

source ~/.bashrc
conda activate job_new

FOLDS=${FOLDS:-"0 1 2 3 4"}
EPOCHS=${EPOCHS:-20}
IMG_BASE=${IMG_BASE:-data/informative_frames}
SPLITS_DIR=${SPLITS_DIR:-informative_frames}
OUT_DIR=${OUT_DIR:-informative_frames/models}
PATIENCE=${PATIENCE:-3}   # early stopping: epochs without val PR-AUC improvement before stopping
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-8}

for fold in $FOLDS; do
  echo "=== train classifier fold=$fold epochs=$EPOCHS ==="
  python -u -m informative_frames.classify_frames \
    --fold "$fold" --epochs "$EPOCHS" --patience "$PATIENCE" \
    --img_base "$IMG_BASE" --split_dir "$SPLITS_DIR" --models_dir "$OUT_DIR" \
    --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS"
done

echo "All classifier training runs finished. Models in $OUT_DIR"
