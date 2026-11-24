#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40medium
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# MONAI version of scripts/preprocess_all.sh (additive; the original
# cropping//informative_frames pipeline is untouched). Same 8 stages, but the
# models/transforms come from the monai_preproc package with a configurable
# experiment grid (see monai_preproc/configs.py) to find the best
# cost/performance balance.
#
# Usage: sbatch scripts/monai_preprocess_all.sh
#        CONFIG=unet_small_384 EPOCHS_CROP=10 EPOCHS_CLS=5 sbatch scripts/monai_preprocess_all.sh

set -e
source ~/.bashrc
conda activate job_new
cd "${SLURM_SUBMIT_DIR:-.}"

CONFIG=${CONFIG:-unet_medium_512}
EPOCHS_CROP=${EPOCHS_CROP:-20}
EPOCHS_CLS=${EPOCHS_CLS:-15}
FOLDS=${FOLDS:-"0 1 2 3 4"}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-8}
OUT_BASE=${OUT_BASE:-data/ensemble_results_paxos2025_monai}
SEG_MODELS=${SEG_MODELS:-monai_preproc/models/seg}
CLS_MODELS=${CLS_MODELS:-monai_preproc/models/cls}

echo "=== 1/8 Split creation (q5_v3-based, patient-level; shared with the existing pipeline) ==="
python -u -m cropping.make_splits_from_q5v3

echo "=== 2/8 Label distribution plots ==="
python -u -m informative_frames.analyze_labels \
  informative_frames/split_info_5fold_paxos2025_from_q5v3.csv \
  --outdir informative_frames/label_plots

echo "=== 3/8 MONAI cropping (segmentation) training, $EPOCHS_CROP epochs per fold ==="
for fold in $FOLDS; do
  python -u -m monai_preproc.train seg --config "$CONFIG" --fold "$fold" \
    --epochs "$EPOCHS_CROP" --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --out_dir "$SEG_MODELS"
done

echo "=== 4/8 MONAI cropping evaluation (metrics + plot + cost) ==="
python -u -m monai_preproc.evaluate seg --config "$CONFIG" \
  --models_dir "$SEG_MODELS" --out_dir "monai_preproc/eval/seg/$CONFIG" --batch_size "$BATCH_SIZE"

echo "=== 5/8 Cropping of the labeled informative frames (existing stage, fold models of the val fold) ==="
python -u -m informative_frames.crop_frames

echo "=== 6/8 MONAI informative-frame classifier training, $EPOCHS_CLS epochs per fold ==="
for fold in $FOLDS; do
  python -u -m monai_preproc.train cls --config "$CONFIG" --fold "$fold" \
    --epochs "$EPOCHS_CLS" --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --out_dir "$CLS_MODELS"
done

echo "=== 7/8 MONAI classifier evaluation (metrics + plot + cost) ==="
python -u -m monai_preproc.evaluate cls --config "$CONFIG" \
  --models_dir "$CLS_MODELS" --out_dir "monai_preproc/eval/cls/$CONFIG"

echo "=== 8/8 Cleaned-video extraction (q5_v3 videos present locally; missing ones are skipped with a warning) ==="
python -u -m monai_preproc.extract \
  --video_dir data/dr_videos \
  --seg_model_dir "$SEG_MODELS" --cls_model_dir "$CLS_MODELS" \
  --config "$CONFIG" \
  --out_base "$OUT_BASE"

echo "MONAI pre-processing finished. Outputs:"
echo "  monai_preproc/eval/seg/$CONFIG/   cropping metrics + plot + cost"
echo "  monai_preproc/eval/cls/$CONFIG/   classifier metrics + plot + cost"
echo "  $OUT_BASE/cleaned_videos/         CLEAN_<video>.mp4 (same naming as the existing pipeline)"
echo "  $OUT_BASE/txt_files/, $OUT_BASE/plots/"
