#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Per-video attention plots of trained DINO transformer runs. RUNS lists one run dir per split 1..N (indexed by split
# number, so SPLITS="3" needs at least 3 entries).
# Usage (from scripts/): RUNS="transformer/models/<run of split 1> ..." SPLITS="1" sbatch vis_attention.sh
#   SP= split dir the runs were trained on, OUT_DIR= output dir, LIMIT= videos per split.
SP=${SP:-data/stratified_splits_paxos2025_patient_q5_v3}
[ -n "$RUNS" ] || { echo "set RUNS=<transformer run dirs, one per split>"; exit 1; }

source ~/.bashrc
conda activate job_new
cd ..

python -u -m dino.vis_attention \
  --trans_paths $RUNS \
  --splits ${SPLITS:-1} \
  --annotations_path "$SP" \
  --complex_augs \
  ${LIMIT:+--limit $LIMIT} \
  --output_dir "${OUT_DIR:-transformer_attention_analysis}"
