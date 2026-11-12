#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Frame-level DINOv3 classifier (5-class, unfrozen backbone), one split; same config as the `clf` job of train_pgcw.sh.
# Usage (from scripts/): SPLIT=1 sbatch train_dino_classifier.sh   (SP= other split dir, TAG= run tag, EXTRA= extra args)
SP=${SP:-data/stratified_splits_paxos2025_patient_q5_v3}
SPLIT=${SPLIT:-1}
TAG=${TAG:-pgcw}

source ~/.bashrc
conda activate job_new
cd ..

python -u -m dino.train_dino_classifier \
  --split_path "$SP/split_$SPLIT" \
  --lr 5e-6 --batch_size 2 --epochs 15 --lr_schedule cosine --min_lr 1e-7 --weight_decay 0.0 \
  --img_size 512 --complex_augs --distance_scale_weight 0.2 \
  --sampler_power 0.5 --class_weight_power 0.5 \
  --run_tag "clf_$TAG" $EXTRA
