#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40medium
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Cost/performance sweep over the MONAI segmentation configs (seg stage only;
# the informative-frame classifier is a separate pipeline, added later).
# Default: the "arch" preset (configs.py ARCH_GRID) -- a grid search over UNet
# (MONAI FlexibleUNet) vs UNet++ (smp UnetPlusPlus), both with ImageNet-
# pretrained EfficientNet encoders, across backbone x input-size combos.
# Trains + evaluates every config on the leakage-free q5_v3 val folds and writes
# $OUT_DIR/sweep_results.csv + pareto.csv + overview.png (perf-vs-cost plot).
# Training logs go to TensorBoard under $OUT_DIR/models/<config>/seg/runs/.
#
# Usage: sbatch scripts/monai_sweep.sh                        # arch grid, folds 0 1 (quick first pass)
#        FOLDS="0 1 2 3 4" sbatch scripts/monai_sweep.sh      # FINAL RUN: all 5 folds
#        ONLY="unet_b1_512,unetpp_b1_512" EPOCHS=10 sbatch scripts/monai_sweep.sh
#        NO_TRAIN=1 sbatch scripts/monai_sweep.sh             # re-evaluate existing checkpoints only
#        PATIENCE=0 sbatch scripts/monai_sweep.sh             # disable early stopping (default: 3)
#        PRESET=default sbatch scripts/monai_sweep.sh         # original mixed grid (reference)
# Re-render pareto.csv + overview.png from an existing CSV without training:
#        python -u -m monai_preproc.sweep --plot_only results/monai_sweep/sweep_results.csv

set -e
source ~/.bashrc
conda activate job_new
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR=${OUT_DIR:-results/monai_sweep}
PRESET=${PRESET:-arch}
FOLDS=${FOLDS:-"0 1"}          # quick first pass; use FOLDS="0 1 2 3 4" for the final full run
GRID_JSON=${GRID_JSON:-}
ONLY=${ONLY:-}
EPOCHS=${EPOCHS:-40}
PATIENCE=${PATIENCE:-}
NUM_WORKERS=${NUM_WORKERS:-8}

CMD=(python -u -m monai_preproc.sweep --out_dir "$OUT_DIR" --num_workers "$NUM_WORKERS"
     --preset "$PRESET" --epochs "$EPOCHS" --folds $FOLDS)
if [ -n "$GRID_JSON" ]; then CMD+=(--grid "$GRID_JSON"); fi
if [ -n "$PATIENCE" ]; then CMD+=(--patience "$PATIENCE"); fi
if [ -n "$ONLY" ]; then CMD+=(--only "$ONLY"); fi
if [ "$NO_TRAIN" = "1" ]; then CMD+=(--no-train); fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
