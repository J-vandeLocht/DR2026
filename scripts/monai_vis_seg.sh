#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40medium
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Visualize seg predictions of a swept config on val images (per fold: input |
# GT mask + GT ellipse | pred mask + fitted pred ellipse | TP/FP/FN), using the
# fold checkpoints under results/monai_sweep/models/<config>/seg.
# Writes $OUT_DIR/<config>_fold<i>_val.png + _val_metrics.csv.
#
# Usage: sbatch scripts/monai_vis_seg.sh
#        CONFIG=flexunet_b0_256_shape100 FOLDS="0 1" N_IMAGES=6 sbatch scripts/monai_vis_seg.sh
#        MODELS_ROOT=results/monai_sweep_b0 OUT_DIR=results/monai_sweep_b0/vis sbatch scripts/monai_vis_seg.sh

set -e
source ~/.bashrc
conda activate job_new
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CONFIG=${CONFIG:-flexunet_b0_256}
PRESET=${PRESET:-flexunet}
FOLDS=${FOLDS:-0}
N_IMAGES=${N_IMAGES:-4}
MODELS_ROOT=${MODELS_ROOT:-results/monai_sweep/models}
OUT_DIR=${OUT_DIR:-results/monai_sweep/vis}

CMD=(python -u -m monai_preproc.vis_seg --config "$CONFIG" --preset "$PRESET"
     --models_root "$MODELS_ROOT" --out_dir "$OUT_DIR" --folds $FOLDS --n_images "$N_IMAGES")

echo "Running: ${CMD[*]}"
"${CMD[@]}"
