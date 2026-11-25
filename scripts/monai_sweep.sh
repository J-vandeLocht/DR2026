#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40medium
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Cost/performance sweep over the MONAI pre-processing configs
# (monai_preproc/configs.py DEFAULT_GRID, optionally extended with GRID_JSON=...).
# Trains + evaluates every config on the leakage-free q5_v3 val folds and writes
# results/monai_sweep/sweep_results.csv + pareto.csv.
#
# Usage: sbatch scripts/monai_sweep.sh
#        ONLY="unet_tiny_256,unet_small_384" EPOCHS_SEG=10 EPOCHS_CLS=5 sbatch scripts/monai_sweep.sh
#        NO_TRAIN=1 sbatch scripts/monai_sweep.sh   (re-evaluate existing checkpoints only)

set -e
source ~/.bashrc
conda activate job_new
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR=${OUT_DIR:-results/monai_sweep}
GRID_JSON=${GRID_JSON:-}
ONLY=${ONLY:-}
EPOCHS_SEG=${EPOCHS_SEG:-40}
EPOCHS_CLS=${EPOCHS_CLS:-10}
NUM_WORKERS=${NUM_WORKERS:-8}

CMD=(python -u -m monai_preproc.sweep --out_dir "$OUT_DIR" --num_workers "$NUM_WORKERS"
     --epochs_seg "$EPOCHS_SEG" --epochs_cls "$EPOCHS_CLS")
if [ -n "$GRID_JSON" ]; then CMD+=(--grid "$GRID_JSON"); fi
if [ -n "$ONLY" ]; then CMD+=(--only "$ONLY"); fi
if [ "$NO_TRAIN" = "1" ]; then CMD+=(--no-train); fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
