#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A100short
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# HISTORICAL: baseline v2 on the old video-level splits (data/stratified_splits_paxos2025) with a classifier checkpoint that
# is not shared. Kept for reference; current runs: train_pgcw.sh / train_dino_transformer.sh (see README.md).
# Baseline v2 (set 2026-10-14, Paxos2025 clips replaced that day): frozen-backbone transformer with cluster
# sampling, lr 1e-5, 10 epochs, cosine LR decay to 1e-7, checkpoint selected by mean(QWK, referable recall)
# (acc added to --selection_metric combined on 2026-10-20; results below used QWK + RR only),
# Paxos2025 split_1, 5-class.
# 2026-10-20: --cluster_sample -> --random_segment_sample (split 1: QWK 0.871 | acc 0.700 | F1 0.629 | RR 0.818,
#   run 2026-10-20_13-32_..._bv2_randseg_ctrl; cluster sampling has no Paxos2020 clusters, so it cannot run with --fused_dataset).
# 2026-10-15: StepLR (x0.3 after epoch 4) -> --lr_schedule cosine. Splits 1-3 mean best QWK 0.864 vs 0.853, RR 0.866 vs 0.846;
#   split 1 cosine: best (epoch 8) QWK 0.848 | acc 0.683 | macro F1 0.646 | referable recall 0.848 (see TODO_experiments.md).
# Changed 2026-10-14 after the reference run: --lr 5e-5 -> 1e-5, train_dino_transformer.py's StepLR gamma
# hardcoded 0.1 -> 0.3, and --selection_metric combined (was QWK; would also have picked epoch 7 there).
# The reference run below used lr 5e-5 / gamma 0.1 and can't be reproduced with this script.
# Reference run: 2026-10-14_13-32_dino_sa_all_split_1_full_cluster_baseline_v2 (seed 42)
#   best (epoch 7, last): QWK 0.862 | acc 0.733 | macro F1 0.693 | referable recall 0.848 | ROC-AUC 0.902
#   epochs 0-4 (lr 5e-5) unstable (QWK 0.66-0.83, referable recall down to 0.545); epochs 5-7 (lr 5e-6) all QWK >= 0.85.
#   All 5 missed referable videos are grade 2 (3 -> grade 0, 2 -> grade 1); grade-2 recall 6/13.
#
# Previous baseline (v1, old clips, --random_segment_sample, 6 epochs, lr_step 3):
#   2026-10-14_08-21_dino_sa_all_split_1_full_baseline, best (epoch 3): QWK 0.850 | acc 0.650 | macro F1 0.593 | referable recall 0.848
#   seed 43, best (epoch 2): QWK 0.881 | acc 0.667 | macro F1 0.581 | referable recall 0.788
#   with --cluster_sample (2026-10-13_12-30_..._cluster_frozen_b2h8_ent0_dist0.3), best (epoch 5): QWK 0.862 | acc 0.717 | macro F1 0.672 | referable recall 0.848

source ~/.bashrc
conda activate job_new
cd ..

python -u -m dino.train_dino_transformer \
  --classifier_checkpoint classifier/models/dino_all_split_1_1002_1543_full/best_model.pth \
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
  --run_tag baseline_v2
