#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40medium
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Cost/performance sweep over the MONAI pre-processing configs.
# Default: the FlexibleUNet-only architecture grid (PRESET=flexunet, configs.py
# FLEXUNET_GRID, optionally extended with GRID_JSON=...). Trains + evaluates the
# selected stage(s) of every config on the leakage-free q5_v3 val folds and writes
# $OUT_DIR/sweep_results.csv + pareto.csv + overview.png (perf-vs-cost plot).
#
# Usage: sbatch scripts/monai_sweep.sh                        # flexunet grid, both stages
#        STAGE=seg sbatch scripts/monai_sweep.sh              # segmentation search only
#        STAGE=cls sbatch scripts/monai_sweep.sh              # classifier search only
#        STAGE=seg SHAPE_LOSS=100 sbatch scripts/monai_sweep.sh  # also run each config with the
#                                                             # ellipse shape-prior loss ('<name>_shape100')
#        ONLY="flexunet_b0_384,flexunet_b1_512" STAGE=seg EPOCHS_SEG=10 sbatch scripts/monai_sweep.sh
#        NO_TRAIN=1 sbatch scripts/monai_sweep.sh             # re-evaluate existing checkpoints only
#        PRESET=default sbatch scripts/monai_sweep.sh         # original mixed UNet/FlexibleUNet grid
# Re-render pareto.csv + overview.png from an existing CSV without training:
#        python -u -m monai_preproc.sweep --plot_only results/monai_sweep/sweep_results.csv

set -e
source ~/.bashrc
conda activate job_new
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR=${OUT_DIR:-results/monai_sweep}
PRESET=${PRESET:-flexunet}
STAGE=${STAGE:-both}
SHAPE_LOSS=${SHAPE_LOSS:-}
GRID_JSON=${GRID_JSON:-}
ONLY=${ONLY:-}
EPOCHS_SEG=${EPOCHS_SEG:-40}
EPOCHS_CLS=${EPOCHS_CLS:-10}
NUM_WORKERS=${NUM_WORKERS:-8}

CMD=(python -u -m monai_preproc.sweep --out_dir "$OUT_DIR" --num_workers "$NUM_WORKERS"
     --preset "$PRESET" --stage "$STAGE"
     --epochs_seg "$EPOCHS_SEG" --epochs_cls "$EPOCHS_CLS")
if [ -n "$GRID_JSON" ]; then CMD+=(--grid "$GRID_JSON"); fi
if [ -n "$SHAPE_LOSS" ]; then CMD+=(--shape_loss "$SHAPE_LOSS"); fi
if [ -n "$ONLY" ]; then CMD+=(--only "$ONLY"); fi
if [ "$NO_TRAIN" = "1" ]; then CMD+=(--no-train); fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
