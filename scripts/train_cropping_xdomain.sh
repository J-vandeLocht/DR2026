#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A100devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# Cross-dataset cropping experiment: train the Unet++ (B4) segmentation model for
# configs 2020 / 2025 / both, each over 5 folds (splits from cropping/make_splits_simple.py).
# Single-dataset runs train for double the epochs of the 'both' runs (the two
# cohorts are about the same size): BASE_EPOCHS for 'both', 2*BASE_EPOCHS otherwise.
#
# Prerequisite (run from the repo root first; on a fresh checkout this also generates
# annotations_cropping_<year>.csv from the video folders, then the 5 subject-level folds):
#   python -u -m cropping.make_splits_simple
#     defaults: --video_dir_2025 data/ensemble_results_paxos2025/cleaned_videos
#               --video_dir_2020 data/ensemble_results_paxos2020/cleaned_videos
# The annotated frames must exist in cropping/frames_to_annotate_larger/.
#
# Usage: sbatch scripts/train_cropping_xdomain.sh
#        CONFIGS="2025" bash scripts/train_cropping_xdomain.sh   (subset of configs)

source ~/.bashrc
conda activate job_new

CONFIGS=${CONFIGS:-"2020 2025 both"}
BASE_EPOCHS=${BASE_EPOCHS:-20}
FOLDS=${FOLDS:-"0 1 2 3 4"}
IMG_DIR=${IMG_DIR:-cropping/frames_to_annotate_larger}
ANN_DIR=${ANN_DIR:-cropping}
OUT_DIR=${OUT_DIR:-cropping/models_xdomain}

for cfg in $CONFIGS; do
  if [ "$cfg" = "both" ]; then epochs=$BASE_EPOCHS; else epochs=$((2 * BASE_EPOCHS)); fi
  for fold in $FOLDS; do
    echo "=== train dataset=$cfg fold=$fold epochs=$epochs ==="
    python -u -m cropping.train_segmentation \
      --dataset "$cfg" --fold "$fold" --epochs "$epochs" \
      --img_dir "$IMG_DIR" --ann_dir "$ANN_DIR" --out_dir "$OUT_DIR"
  done
done

echo "All cropping xdomain training runs finished. Models in $OUT_DIR"
