#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# DINOv3 ConvNeXt-L transformer (stage-3 features, backbone lr x0.1, sparse pooling), one split; same config as the
# `cnxp` job of train_pgcw.sh (MIXED=1: + 53 Paxos2020 training videos = the `cnx` job).
# Usage (from scripts/): SPLIT=1 sbatch train_convnext_transformer.sh   (SP=, TAG=, EXTRA= as in train_dino_classifier.sh)
SP=${SP:-data/stratified_splits_paxos2025_patient_q5_v3}
SPLIT=${SPLIT:-1}
TAG=${TAG:-pgcw}
if [ -n "$MIXED" ]; then data="--train_json mil_train_mixed.json --fused_dataset"; name=cnx_sparse_mixed_$TAG
else data=""; name=cnx_sparse_$TAG; fi

source ~/.bashrc
conda activate job_new
cd ..

python -u -m convnext.train_convnext_transformer \
  --split_path "$SP/split_$SPLIT" $data \
  --adapter linear_attnpool --feature_stage 3 --feature_block -1 \
  --epochs 25 --lr 1e-5 --min_lr 1e-7 --weight_decay 0.0 --backbone_lr_mult 0.1 \
  --img_size 512 --complex_augs --num_frames 32 --random_segment_sample \
  --aggregation_mode sparse_pool --num_query_tokens 4 --num_blocks 1 --num_heads 8 --ffn_dim 2048 --dropout 0.1 \
  --distance_scale_weight 0.2 --selection_metric combined \
  --sampler_power 0.5 --class_weight_power 0.5 \
  --run_tag "$name" $EXTRA
