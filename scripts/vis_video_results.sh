#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

source ~/.bashrc
conda activate job_new
cd ..

CLASSIFIER_PATHS_BINARY=(
"classifier/models/dino_binary_split_1_0906_2310_full/best_model.pth"
"classifier/models/dino_binary_split_2_0906_2340_full/best_model.pth"
"classifier/models/dino_binary_split_3_0907_0010_full/best_model.pth"
"classifier/models/dino_binary_split_4_0906_2208_full/best_model.pth"
"classifier/models/dino_binary_split_5_0906_2239_full/best_model.pth"
)

CLASSIFIER_PATHS_ALL=(
"classifier/models/dino_all_split_1_0906_1934_full/best_model.pth"
"classifier/models/dino_all_split_2_0906_2005_full/best_model.pth"
"classifier/models/dino_all_split_3_0906_2036_full/best_model.pth"
"classifier/models/dino_all_split_4_0906_2107_full/best_model.pth"
"classifier/models/dino_all_split_5_0906_2138_full/best_model.pth"
)

TRANSFORMER_PATHS_BINARY=(
"transformer/models/dino_sa_binary_split_1_0907_1613_full/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_2_0907_1524_full/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_3_0907_1500_full/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_4_0907_1442_full/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_5_0907_1352_full/best_sa_model.pth"
)

TRANSFORMER_PATHS_BINARY_2L=(
"transformer/models/dino_sa_binary_split_1_0925_1835_full_2blocks/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_2_0925_1901_full_2blocks/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_3_0925_2006_full_2blocks/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_4_0925_2133_full_2blocks/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_5_0926_0135_full_2blocks/best_sa_model.pth"
)

TRANSFORMER_PATHS_ALL=(
"transformer/models/dino_sa_all_split_1_0907_1147_full/best_sa_model.pth"
"transformer/models/dino_sa_all_split_2_0907_1203_full/best_sa_model.pth"
"transformer/models/dino_sa_all_split_3_0907_1224_full/best_sa_model.pth"
"transformer/models/dino_sa_all_split_4_0907_1314_full/best_sa_model.pth"
"transformer/models/dino_sa_all_split_5_0907_1330_full/best_sa_model.pth"
)

TRANSFORMER_PATHS_ALL_2L=(
"transformer/models/dino_sa_all_split_1_0926_1618_full_2blocks/best_sa_model.pth"
"transformer/models/dino_sa_all_split_2_0926_1856_full_2blocks/best_sa_model.pth"
"transformer/models/dino_sa_all_split_3_0926_2123_full_2blocks/best_sa_model.pth"
"transformer/models/dino_sa_all_split_4_0928_1849_full_2blocks/best_sa_model.pth"
"transformer/models/dino_sa_all_split_5_0928_2110_full_2blocks/best_sa_model.pth"
)

#python -u -m dino.vis_video_results \
#  --dataset_name "2025" \
#  --annotations_path "data/stratified_splits" \
#  --videos_path "data/ensemble_results_paxos2025/cleaned_videos" \
#  --output_dir "data/multi_eval_paxos2025_binary_new" \
#  --model Clf classifier "${CLASSIFIER_PATHS_BINARY[@]}" \
#  --model Clf32  classifier_video "${CLASSIFIER_PATHS_BINARY[@]}" \
#  --model Trans_1L transformer "${TRANSFORMER_PATHS_BINARY[@]}" \
#  --model Trans_2L transformer "${TRANSFORMER_PATHS_BINARY_2L[@]}" \
#  --img_size 512 \
#  --complex_augs \
#  --binary_classification

#python -u -m dino.vis_video_results \
#  --dataset_name "2025" \
#  --annotations_path "data/stratified_splits" \
#  --videos_path "data/ensemble_results_paxos2025/cleaned_videos" \
#  --output_dir "data/multi_eval_paxos2025_all_2" \
#  --model Clf classifier "${CLASSIFIER_PATHS_ALL[@]}" \
#  --model Clf32  classifier_video "${CLASSIFIER_PATHS_ALL[@]}" \
#  --model Trans_1L transformer "${TRANSFORMER_PATHS_ALL[@]}" \
#  --model Trans_2L transformer "${TRANSFORMER_PATHS_ALL_2L[@]}" \
#  --img_size 512 \
#  --complex_augs

# 5-split baselines (2026-10-21, new clips, data/stratified_splits_paxos2025): classifier ensembles vs. DINO baseline v2
# vs. ConvNeXt baseline. Classifier for split i = the checkpoint that initialized the DINO transformer of split i.
# Analysis: cd data/multi_eval_paxos2025_all && python analyse_runs.py --input_dir ../multi_eval_paxos2025_baselines \
#   --output_dir ../multi_eval_paxos2025_baselines/split_analysis_results
CLASSIFIER_PATHS_V2=(
"classifier/models/dino_all_split_1_1002_1543_full/best_model.pth"
"classifier/models/dino_all_split_2_1002_1543_full/best_model.pth"
"classifier/models/dino_all_split_3_1002_1543_full/best_model.pth"
"classifier/models/dino_all_split_4_1005_0857_full/best_model.pth"
"classifier/models/dino_all_split_5_1005_0855_full/best_model.pth"
)

DINO_BV2_PATHS=(
"transformer/models/2026-10-20_13-32_dino_sa_all_split_1_full_sel-combined_bv2_randseg_ctrl/best_sa_model.pth"
"transformer/models/2026-10-20_18-01_dino_sa_all_split_2_full_sel-combined_baseline_v2/best_sa_model.pth"
"transformer/models/2026-10-20_18-03_dino_sa_all_split_3_full_sel-combined_baseline_v2/best_sa_model.pth"
"transformer/models/2026-10-21_08-40_dino_sa_all_split_4_full_sel-combined_baseline_v2/best_sa_model.pth"
"transformer/models/2026-10-21_11-36_dino_sa_all_split_5_full_sel-combined_baseline_v2/best_sa_model.pth"
)

# Classifiers retrained 2026-10-22 with --lr_schedule cosine --distance_scale_weight 0.2 (bs 2, otherwise as V2).
CLASSIFIER_PATHS_COS=(
"classifier/models/2026-10-22_11-23_dino_all_split_1_full_cos_cdw02_bs2/best_model.pth"
"classifier/models/2026-10-22_11-23_dino_all_split_2_full_cos_cdw02_bs2/best_model.pth"
"classifier/models/2026-10-22_11-23_dino_all_split_3_full_cos_cdw02_bs2/best_model.pth"
"classifier/models/2026-10-22_11-23_dino_all_split_4_full_cos_cdw02_bs2/best_model.pth"
"classifier/models/2026-10-22_11-23_dino_all_split_5_full_cos_cdw02_bs2/best_model.pth"
)
CNX_BASELINE_PATHS=(
"convnext/models/2026-10-20_21-50_convnext_sa_all_split_1_linear_attnpool_s3_sel-combined_cnx_baseline/best_sa_model.pth"
"convnext/models/2026-10-20_21-54_convnext_sa_all_split_2_linear_attnpool_s3_sel-combined_cnx_baseline/best_sa_model.pth"
"convnext/models/2026-10-21_00-49_convnext_sa_all_split_3_linear_attnpool_s3_sel-combined_cnx_baseline/best_sa_model.pth"
"convnext/models/2026-10-21_03-09_convnext_sa_all_split_4_linear_attnpool_s3_sel-combined_cnx_baseline/best_sa_model.pth"
"convnext/models/2026-10-21_08-33_convnext_sa_all_split_5_linear_attnpool_s3_sel-combined_cnx_baseline/best_sa_model.pth"
)

SPLITS=(${SPLITS:-1 2 3 4 5})
python -u -m dino.vis_video_results \
  --dataset_name "2025" \
  --splits "${SPLITS[@]}" \
  --annotations_path "data/stratified_splits_paxos2025" \
  --frame_annotations_path "data/stratified_splits_paxos2025" \
  --videos_path "data/ensemble_results_paxos2025/cleaned_videos" \
  --output_dir "${OUT_DIR:-data/multi_eval_paxos2025_baselines}" \
  --model Clf classifier $(for s in "${SPLITS[@]}"; do echo "${CLASSIFIER_PATHS_V2[$((s-1))]}"; done) \
  --model Clf32 classifier_video $(for s in "${SPLITS[@]}"; do echo "${CLASSIFIER_PATHS_V2[$((s-1))]}"; done) \
  --model ClfCos classifier $(for s in "${SPLITS[@]}"; do echo "${CLASSIFIER_PATHS_COS[$((s-1))]}"; done) \
  --model ClfCos32 classifier_video $(for s in "${SPLITS[@]}"; do echo "${CLASSIFIER_PATHS_COS[$((s-1))]}"; done) \
  --model DINO_Trans transformer $(for s in "${SPLITS[@]}"; do echo "${DINO_BV2_PATHS[$((s-1))]}"; done) \
  --model CNX_Trans convnext_transformer $(for s in "${SPLITS[@]}"; do echo "${CNX_BASELINE_PATHS[$((s-1))]}"; done) \
  --img_size 512 \
  --complex_augs
