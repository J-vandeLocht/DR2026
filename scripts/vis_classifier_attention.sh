#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Occlusion heatmaps of a frame classifier on the cropped Paxos2025 stills.
# Usage (from scripts/): CLF=classifier/models/<run>/best_model.pth sbatch vis_classifier_attention.sh
[ -n "$CLF" ] || { echo "set CLF=<classifier best_model.pth>"; exit 1; }

source ~/.bashrc
conda activate job_new
cd ..

python -u -m dino.vis_classifier_attention \
  --input_dir "${INPUT_DIR:-data/2024_Paxos_Frames/cropped_matched_frames}" \
  --weights "$CLF" \
  --output_dir "${OUT_DIR:-occlusion_results}" \
  --complex_augs
