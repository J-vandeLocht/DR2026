#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A100short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# Crop + filter the raw Paxos2020 videos (the cropping ensemble covers both cohorts).
# VIDEO_DIR defaults to ~/Data/2020_Paxos_Videos/data; shard over several jobs with GPU_ID=i TOTAL_GPUS=n.
VIDEO_DIR=${VIDEO_DIR:-$HOME/Data/2020_Paxos_Videos/data}

source ~/.bashrc
conda activate job_new
cd ..
python -u -m informative_frames.extract_inf_frames \
  --gpu_id "${GPU_ID:-0}" \
  --total_gpus "${TOTAL_GPUS:-1}" \
  --video_dir "$VIDEO_DIR" \
  --seg_model_dir "cropping/models" \
  --cls_model_dir "informative_frames/models" \
  --out_base "data/ensemble_results_paxos2020"
