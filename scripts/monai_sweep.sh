#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40medium
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Cost/performance sweep over the MONAI segmentation configs (seg stage only;
# the informative-frame classifier is a separate pipeline, added later).
# Default: the FlexibleUNet-only grid (PRESET=flexunet, configs.py FLEXUNET_GRID,
# optionally extended with GRID_JSON=...). Trains + evaluates every config on the
# leakage-free q5_v3 val folds and writes $OUT_DIR/sweep_results.csv + pareto.csv +
# overview.png (perf-vs-cost plot). Training logs go to TensorBoard under
# $OUT_DIR/models/<config>/seg/runs/.
#
# Usage: sbatch scripts/monai_sweep.sh                        # flexunet grid
#        ONLY="flexunet_b0_384,flexunet_b1_512" EPOCHS=10 sbatch scripts/monai_sweep.sh
#        SHAPE_LOSS=100 sbatch scripts/monai_sweep.sh         # also run '<name>_shape100'
#                                                             # (Dice + 100*ellipse loss)
#        SHAPE_LOSS=100 ONLY_SHAPE=1 sbatch scripts/monai_sweep.sh   # only the '_shape100' variants
#        NO_TRAIN=1 sbatch scripts/monai_sweep.sh             # re-evaluate existing checkpoints only
#        PATIENCE=0 sbatch scripts/monai_sweep.sh             # disable early stopping (default: 3)
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
SHAPE_LOSS=${SHAPE_LOSS:-}
ONLY_SHAPE=${ONLY_SHAPE:-}
GRID_JSON=${GRID_JSON:-}
ONLY=${ONLY:-}
EPOCHS=${EPOCHS:-40}
PATIENCE=${PATIENCE:-}
NUM_WORKERS=${NUM_WORKERS:-8}

CMD=(python -u -m monai_preproc.sweep --out_dir "$OUT_DIR" --num_workers "$NUM_WORKERS"
     --preset "$PRESET" --epochs "$EPOCHS")
if [ -n "$PATIENCE" ]; then CMD+=(--patience "$PATIENCE"); fi
if [ -n "$GRID_JSON" ]; then CMD+=(--grid "$GRID_JSON"); fi
if [ -n "$SHAPE_LOSS" ]; then CMD+=(--shape_loss "$SHAPE_LOSS"); fi
if [ "$ONLY_SHAPE" = "1" ]; then CMD+=(--only_shape_loss); fi
if [ -n "$ONLY" ]; then CMD+=(--only "$ONLY"); fi
if [ "$NO_TRAIN" = "1" ]; then CMD+=(--no-train); fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
