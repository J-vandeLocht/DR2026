#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A100short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# DINOv3 transformer aggregator (5-class, sparse pooling), one split, initialized from a trained classifier
# (train_dino_classifier.sh); same configs as the `dfz` (frozen) / `dft` (fine-tuned) jobs of train_pgcw.sh.
# Usage (from scripts/): CLF=classifier/models/<run>/best_model.pth SPLIT=1 sbatch train_dino_transformer.sh
#   FT=1 fine-tunes the backbone (lr x0.1, 15 epochs) instead of freezing it (8 epochs). SP=, TAG=, EXTRA= as in train_dino_classifier.sh.
SP=${SP:-data/stratified_splits_paxos2025_patient_q5_v3}
SPLIT=${SPLIT:-1}
TAG=${TAG:-pgcw}
[ -n "$CLF" ] || { echo "set CLF=<classifier best_model.pth>"; exit 1; }
if [ -n "$FT" ]; then mode="--backbone_lr_mult 0.1 --epochs 15"; name=dino_ft_sparse_$TAG
else mode="--freeze_backbone --epochs 8"; name=dino_sparse_$TAG; fi

source ~/.bashrc
conda activate job_new
cd ..

python -u -m dino.train_dino_transformer \
  --classifier_checkpoint "$CLF" \
  --split_path "$SP/split_$SPLIT" \
  --lr 1e-5 --lr_schedule cosine --min_lr 1e-7 --weight_decay 0.0 \
  --img_size 512 --complex_augs --num_frames 32 --random_segment_sample \
  --aggregation_mode sparse_pool --num_query_tokens 4 --num_blocks 2 --num_heads 8 --ffn_dim 2048 --dropout 0.1 \
  --attn_entropy_weight 0.0 --distance_scale_weight 0.3 --selection_metric combined \
  --sampler_power 0.5 --class_weight_power 0.5 \
  $mode --run_tag "$name" $EXTRA
