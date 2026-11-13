#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A100short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Production extraction of informative frames per video with the x-domain models.
# Per video (rule in informative_frames/extract_inf_frames.py, same as
# cropping/evaluate_segmentation.py): if the model config matches the video's dataset,
# ONLY the fold model whose val split (cropping splits_<dataset>.json) contains the
# video is used -- the model that was NOT trained on it; otherwise (other dataset, or
# video unknown to the splits) the full 5-fold mean-sigmoid-probability ensemble.
#
# Prerequisites: trained x-domain models
#   cropping/models_xdomain/seg_<cfg>_fold_<i>.pth        (shared dir; matched per config)
#   informative_frames/models_xdomain/cls_<cfg>_fold_<i>.pth
#
# The three runs (env vars, then sbatch):
#
# 1) Paxos2025 videos with the Paxos2025 models (per-video val-fold model):
#      DATASET=2025 VIDEO_DIR=data/dr_videos \
#      SEG_MODEL_DIR=cropping/models_xdomain CLS_MODEL_DIR=informative_frames/models_xdomain \
#      SEG_CONFIG=2025 CLS_CONFIG=2025 OUT_BASE=data/ensemble_results_paxos2025_by_2025 \
#      sbatch scripts/extract_inf_frames_xdomain.sh
#
# 2) Paxos2020 videos with the Paxos2020 models (per-video val-fold model):
#      DATASET=2020 VIDEO_DIR=experiments/Paxos_2020/data \
#      SEG_MODEL_DIR=cropping/models_xdomain CLS_MODEL_DIR=informative_frames/models_xdomain \
#      SEG_CONFIG=2020 CLS_CONFIG=2020 \
#      OUT_BASE=experiments/Paxos_2020/data/ensemble_results_paxos2020_by_2020 \
#      sbatch scripts/extract_inf_frames_xdomain.sh
#
# 3) Paxos2020 videos with the Paxos2025 models (cross-dataset -> 5-fold ensemble):
#      DATASET=2020 VIDEO_DIR=experiments/Paxos_2020/data \
#      SEG_MODEL_DIR=cropping/models_xdomain CLS_MODEL_DIR=informative_frames/models_xdomain \
#      SEG_CONFIG=2025 CLS_CONFIG=2025 \
#      OUT_BASE=experiments/Paxos_2020/data/ensemble_results_paxos2020_by_2025 \
#      sbatch scripts/extract_inf_frames_xdomain.sh
#
# Multi-GPU: raise TOTAL_GPUS and launch one job per GPU_ID 0..TOTAL_GPUS-1.

source ~/.bashrc
conda activate job_new

DATASET=${DATASET:?set DATASET=2025 or 2020}
VIDEO_DIR=${VIDEO_DIR:?set VIDEO_DIR}
SEG_MODEL_DIR=${SEG_MODEL_DIR:?set SEG_MODEL_DIR}
CLS_MODEL_DIR=${CLS_MODEL_DIR:?set CLS_MODEL_DIR}
SEG_CONFIG=${SEG_CONFIG:?set SEG_CONFIG}
CLS_CONFIG=${CLS_CONFIG:?set CLS_CONFIG}
OUT_BASE=${OUT_BASE:?set OUT_BASE}
GPU_ID=${GPU_ID:-0}
TOTAL_GPUS=${TOTAL_GPUS:-1}
SPLITS_DIR=${SPLITS_DIR:-cropping}

python -u -m informative_frames.extract_inf_frames \
  --gpu_id "$GPU_ID" --total_gpus "$TOTAL_GPUS" \
  --video_dir "$VIDEO_DIR" \
  --seg_model_dir "$SEG_MODEL_DIR" --cls_model_dir "$CLS_MODEL_DIR" \
  --out_base "$OUT_BASE" \
  --dataset "$DATASET" --seg_config "$SEG_CONFIG" --cls_config "$CLS_CONFIG" \
  --splits_dir "$SPLITS_DIR"
