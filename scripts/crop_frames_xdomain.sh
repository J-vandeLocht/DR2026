#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Crop the raw informative frames of BOTH datasets with EACH x-domain cropping model
# config -> 6 output folders (logic in informative_frames/crop_frames.py):
#   data/informative_frames/frames_cropped_paxos<ds>_by_<cfg>/{0,1}/
#     ds in {2025, 2020}, cfg in {2020, 2025, both}
# Same rule as cropping/evaluate_segmentation.py: in-dataset (cfg == ds or cfg == both)
# -> each frame is cropped by the fold-i model of its val fold (never trained on it,
# fold from the split CSV); cross-dataset -> 5-fold mean-sigmoid-probability ensemble.
#
# Prerequisites (run from the repo root first):
#   python -u -m informative_frames.make_splits_from_cropping   # split CSVs with label + val fold
#   trained checkpoints in cropping/models/{2020,2025,both} (*fold_<i>* glob)
#   raw frames in data/informative_frames/frames_raw_extract{,_paxos2020}
#
# Usage: sbatch scripts/crop_frames_xdomain.sh
#        DATASETS="2025" CONFIGS="2020 both" bash scripts/crop_frames_xdomain.sh   (subset)

source ~/.bashrc
conda activate job_new

DATASETS=${DATASETS:-"2025 2020"}
CONFIGS=${CONFIGS:-"2020 2025 both"}
IMG_BASE=${IMG_BASE:-data/informative_frames}
MODELS_BASE=${MODELS_BASE:-cropping/models}
SPLITS_DIR=${SPLITS_DIR:-informative_frames}
BATCH_SIZE=${BATCH_SIZE:-16}

for ds in $DATASETS; do
  case $ds in
    2025) IN_DIR=$IMG_BASE/frames_raw_extract ;;
    2020) IN_DIR=$IMG_BASE/frames_raw_extract_paxos2020 ;;
  esac
  SPLIT_CSV=$SPLITS_DIR/split_info_5fold_paxos${ds}_from_cropping.csv
  for cfg in $CONFIGS; do
    OUT_DIR=$IMG_BASE/frames_cropped_paxos${ds}_by_${cfg}
    echo "=== crop dataset=$ds with cropping model config=$cfg -> $OUT_DIR ==="
    python -u -m informative_frames.crop_frames \
      --dataset "$ds" --config "$cfg" \
      --input_dir "$IN_DIR" --split_csv "$SPLIT_CSV" \
      --model_dir "$MODELS_BASE/$cfg" \
      --output_dir "$OUT_DIR" \
      --batch_size "$BATCH_SIZE"
  done
done

echo "All cropping folder runs finished (requested subset of the 6). Folders under $IMG_BASE/frames_cropped_paxos*_by_*"
