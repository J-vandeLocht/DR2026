#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A100short
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# Baseline v2 with dense DINOv3 patch tokens instead of CLS: the 32x32 patch grid goes through an adapter
# (transformer/adapters.py) giving K tokens per frame, and the bag transformer attends over all 32*K tokens.
# Usage: sbatch train_dino_transformer_dense.sh [adapter] [dense_tokens] [extra args...]
#   linear_attnpool 1 (default) | linear_attnpool 4 | linear_mean (grid adapter removed 2026-10-26)
#   extra args e.g. --no_adapter_pe; CLF= (empty) starts from pretrained DINOv3 instead of the classifier.

ADAPTER=${1:-linear_attnpool}
TOKENS=${2:-1}
shift $(( $# < 2 ? $# : 2 ))
CLF=${CLF-classifier/models/dino_all_split_1_1002_1543_full/best_model.pth}
TAG=${TAG:-dense_${ADAPTER}${TOKENS}}

source ~/.bashrc
conda activate job_new
cd ..

python -u -m dino.train_dino_transformer \
  ${CLF:+--classifier_checkpoint "$CLF"} \
  --split_path data/stratified_splits_paxos2025/split_1 \
  --train_json mil_train.json --val_json mil_val.json \
  --seed 42 \
  --freeze_backbone \
  --epochs 10 --lr 1e-5 --lr_schedule cosine --min_lr 1e-7 \
  --img_size 512 --complex_augs \
  --num_blocks 2 --num_heads 8 --ffn_dim 2048 --dropout 0.1 --num_frames 32 \
  --attn_entropy_weight 0.0 --distance_scale_weight 0.3 \
  --random_segment_sample \
  --selection_metric combined \
  --dense_adapter "$ADAPTER" --dense_tokens "$TOKENS" \
  --run_tag "$TAG" \
  "$@"
