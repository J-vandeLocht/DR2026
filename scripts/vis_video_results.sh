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
#python -u -m dino.vis_video_results \
#  --dataset_name "2025" \
#  --splits "${SPLITS[@]}" \
#  --annotations_path "data/stratified_splits_paxos2025" \
#  --frame_annotations_path "data/stratified_splits_paxos2025" \
#  --videos_path "data/ensemble_results_paxos2025/cleaned_videos" \
#  --output_dir "${OUT_DIR:-data/multi_eval_paxos2025_baselines}" \
#  --model Clf classifier $(for s in "${SPLITS[@]}"; do echo "${CLASSIFIER_PATHS_V2[$((s-1))]}"; done) \
#  --model Clf32 classifier_video $(for s in "${SPLITS[@]}"; do echo "${CLASSIFIER_PATHS_V2[$((s-1))]}"; done) \
#  --model ClfCos classifier $(for s in "${SPLITS[@]}"; do echo "${CLASSIFIER_PATHS_COS[$((s-1))]}"; done) \
#  --model ClfCos32 classifier_video $(for s in "${SPLITS[@]}"; do echo "${CLASSIFIER_PATHS_COS[$((s-1))]}"; done) \
#  --model DINO_Trans transformer $(for s in "${SPLITS[@]}"; do echo "${DINO_BV2_PATHS[$((s-1))]}"; done) \
#  --model CNX_Trans convnext_transformer $(for s in "${SPLITS[@]}"; do echo "${CNX_BASELINE_PATHS[$((s-1))]}"; done) \
#  --img_size 512 \
#  --complex_augs

# Patient-level q5 splits (2026-10-27 *_pgl runs: data/stratified_splits_paxos2025_patient_q5, wd 0), 6 models.
# CKPT=best (default) -> data/multi_eval_paxos2025_patient_q5_best, CKPT=last -> ..._q5_last. Afterwards: referable-DR CSVs
# (make_binary.py), the multiclass + binary reports (analyse_runs.py; F1 @ 0.5, mean +- SE) and pooled ROC curves (plot_roc.py). ~4 min per split on one GPU;
# for one split per GPU run e.g. `SPLITS=3 sbatch vis_video_results.sh` per split, then once with ANALYSE_ONLY=1.
CKPT=${CKPT:-best}
Q5_SPLITS=${Q5_SPLITS:-data/stratified_splits_paxos2025_patient_q5}
Q5_OUT=${OUT_DIR:-data/multi_eval_paxos2025_patient_q5_${CKPT}}
CLF_PGL=(
"classifier/models/2026-10-27_12-39_dino_all_split_1_full_clf_pgl"
"classifier/models/2026-10-27_12-40_dino_all_split_2_full_clf_pgl"
"classifier/models/2026-10-27_12-42_dino_all_split_3_full_clf_pgl"
"classifier/models/2026-10-27_12-43_dino_all_split_4_full_clf_pgl"
"classifier/models/2026-10-27_12-45_dino_all_split_5_full_clf_pgl"
)
DINO_PGL=(
"transformer/models/2026-10-27_22-37_dino_sa_all_split_1_full_sparse_pool_sel-combined_dino_sparse_pgl"
"transformer/models/2026-10-27_22-38_dino_sa_all_split_2_full_sparse_pool_sel-combined_dino_sparse_pgl"
"transformer/models/2026-10-27_22-40_dino_sa_all_split_3_full_sparse_pool_sel-combined_dino_sparse_pgl"
"transformer/models/2026-10-27_22-41_dino_sa_all_split_4_full_sparse_pool_sel-combined_dino_sparse_pgl"
"transformer/models/2026-10-27_23-04_dino_sa_all_split_5_full_sparse_pool_sel-combined_dino_sparse_pgl"
)
DINO_FT_PGL=(
"transformer/models/2026-10-27_19-52_dino_sa_all_split_1_full_sparse_pool_sel-combined_dino_ft_sparse_pgl"
"transformer/models/2026-10-27_19-54_dino_sa_all_split_2_full_sparse_pool_sel-combined_dino_ft_sparse_pgl"
"transformer/models/2026-10-27_19-55_dino_sa_all_split_3_full_sparse_pool_sel-combined_dino_ft_sparse_pgl"
"transformer/models/2026-10-27_19-57_dino_sa_all_split_4_full_sparse_pool_sel-combined_dino_ft_sparse_pgl"
"transformer/models/2026-10-27_20-16_dino_sa_all_split_5_full_sparse_pool_sel-combined_dino_ft_sparse_pgl"
)
CNX_PGL=(
"convnext/models/2026-10-27_16-53_convnext_sa_all_split_1_linear_attnpool_s3_sparse_pool_sel-combined_cnx_sparse_pgl"
"convnext/models/2026-10-27_16-55_convnext_sa_all_split_2_linear_attnpool_s3_sparse_pool_sel-combined_cnx_sparse_pgl"
"convnext/models/2026-10-27_16-57_convnext_sa_all_split_3_linear_attnpool_s3_sparse_pool_sel-combined_cnx_sparse_pgl"
"convnext/models/2026-10-27_16-58_convnext_sa_all_split_4_linear_attnpool_s3_sparse_pool_sel-combined_cnx_sparse_pgl"
"convnext/models/2026-10-27_17-09_convnext_sa_all_split_5_linear_attnpool_s3_sparse_pool_sel-combined_cnx_sparse_pgl"
)
CNX_MIX_PGL=(
"convnext/models/2026-10-27_13-14_convnext_sa_all_split_1_linear_attnpool_s3_sparse_pool_sel-combined_cnx_sparse_mixed_pgl"
"convnext/models/2026-10-27_13-17_convnext_sa_all_split_2_linear_attnpool_s3_sparse_pool_sel-combined_cnx_sparse_mixed_pgl"
"convnext/models/2026-10-27_13-19_convnext_sa_all_split_3_linear_attnpool_s3_sparse_pool_sel-combined_cnx_sparse_mixed_pgl"
"convnext/models/2026-10-27_13-21_convnext_sa_all_split_4_linear_attnpool_s3_sparse_pool_sel-combined_cnx_sparse_mixed_pgl"
"convnext/models/2026-10-27_13-22_convnext_sa_all_split_5_linear_attnpool_s3_sparse_pool_sel-combined_cnx_sparse_mixed_pgl"
)
# <run dir>/<best|last>_<file> for the selected splits
ckpts() { local file=$1; shift; for s in "${SPLITS[@]}"; do local runs=("$@"); echo "${runs[$((s-1))]}/${CKPT}_${file}"; done; }

if [ -n "$XDOM" ]; then
# Cross-domain (scripts/train_xdomain.sh): trained on 100% Paxos2025 q5, tested on 294 filtered Paxos2020 videos, last_*.pth
# only. One pseudo-split per seed (XDOM_SEEDS="42 43 44" -> split_1..3): XDOM=1 sbatch vis_video_results.sh
# CROP=1 for --crop_borders runs (default since 2026-11-01; e.g. XTAG=xdomcw): XDOM=1 XTAG=xdomcw CROP=1 OUT_DIR=data/multi_eval_xdomain_paxos2020_cw sbatch ...
# Without CROP, --no_crop_borders (the old xdom runs). XCNX="<tag> ...": only CNX models, one CNX_<tag> per tag; NOCLAHE=1 -> --no_clahe
# (p20 runs: XDOM=1 XCNX="xdnc xdp20 xdp20c xdp20q xdp20l" NOCLAHE=1 CROP=1 OUT_DIR=data/multi_eval_xdomain_paxos2020_p20).
XTAG=${XTAG:-xdom}
XDOM_SEEDS=(${XDOM_SEEDS:-42})
SPLITS=($(seq 1 ${#XDOM_SEEDS[@]}))
N_CSV=${#XDOM_SEEDS[@]}
TXT_DIR=$(realpath data/ensemble_results_paxos2020/txt_files)
Q5_OUT=${OUT_DIR:-data/multi_eval_xdomain_paxos2020}
xdom() { for seed in "${XDOM_SEEDS[@]}"; do echo "$(ls -d $1/*_$2_${XTAG}_s$seed | tail -1)/last_$3"; done; }
if [ -z "$ANALYSE_ONLY" ]; then
python -u -m dino.vis_video_results \
  --dataset_name "2020" \
  --splits "${SPLITS[@]}" \
  --annotations_path data/stratified_splits_paxos2025_q5_full_xdom \
  --videos_path "data/ensemble_results_paxos2020/cleaned_videos" \
  --output_dir "$Q5_OUT" \
  $(if [ -n "$XCNX" ]; then for t in $XCNX; do echo --model CNX_$t convnext_transformer $(XTAG=$t xdom convnext/models cnx_sparse sa_model.pth); done
    else echo --model Clf32 classifier_video $(xdom classifier/models clf model.pth) \
      --model DINO transformer $(xdom transformer/models dino_sparse sa_model.pth) \
      --model DINO_FT transformer $(xdom transformer/models dino_ft_sparse sa_model.pth) \
      --model CNX convnext_transformer $(xdom convnext/models cnx_sparse sa_model.pth); fi) \
  --img_size 512 \
  --complex_augs $([ -n "$CROP" ] && echo --crop_borders || echo --no_crop_borders) ${NOCLAHE:+--no_clahe}
fi
elif [ -n "$PGCW" ]; then
# *_pgcw (scripts/train_pgcw.sh): *_pgl HPs + --sampler_power 0.5 --class_weight_power 0.5 --crop_borders, so evaluated
# with --crop_borders (all models in the call). Compare with data/multi_eval_paxos2025_patient_q5_best. PGCW=1 sbatch vis_video_results.sh
# Ablations: PGCW=1 PTAG=pgsw PMODELS="clf dft cnx" CROP= sbatch ... (PMODELS: train_pgcw.sh job names; CROP= drops --crop_borders).
PTAG=${PTAG:-pgcw}
# EVAL_ARGS: extra vis_video_results.py args, e.g. EVAL_ARGS="--eval_bags 8 --eval_frames 64" OUT_DIR=... (test-time bag averaging).
PMODELS=" ${PMODELS:-clf dfz dft cnxp cnx} "
CROP=${CROP-1}
Q5_OUT=${OUT_DIR:-data/multi_eval_paxos2025_patient_q5_${PTAG}_${CKPT}}
# newest run dir of split s ending in _<tag> -> <dir>/<best|last>_<file>
pgcw() { for s in "${SPLITS[@]}"; do echo "$(ls -d $1/*_split_${s}_*_$2 | tail -1)/${CKPT}_$3"; done; }
if [ -z "$ANALYSE_ONLY" ]; then
python -u -m dino.vis_video_results \
  --dataset_name "2025" \
  --splits "${SPLITS[@]}" \
  --annotations_path "$Q5_SPLITS" \
  --frame_annotations_path "$Q5_SPLITS" \
  --videos_path "data/ensemble_results_paxos2025/cleaned_videos" \
  --output_dir "$Q5_OUT" \
  $([[ $PMODELS == *" clf "* ]] && echo --model Clf classifier $(pgcw classifier/models clf_$PTAG model.pth) \
     --model Clf32 classifier_video $(pgcw classifier/models clf_$PTAG model.pth)) \
  $([[ $PMODELS == *" dfz "* ]] && echo --model DINO transformer $(pgcw transformer/models dino_sparse_$PTAG sa_model.pth)) \
  $([[ $PMODELS == *" dft "* ]] && echo --model DINO_FT transformer $(pgcw transformer/models dino_ft_sparse_$PTAG sa_model.pth)) \
  $([[ $PMODELS == *" cnxp "* ]] && echo --model CNX convnext_transformer $(pgcw convnext/models cnx_sparse_$PTAG sa_model.pth)) \
  $([[ $PMODELS == *" cnx "* ]] && echo --model CNX_mix convnext_transformer $(pgcw convnext/models cnx_sparse_mixed_$PTAG sa_model.pth)) \
  --img_size 512 \
  --complex_augs $([ -n "$CROP" ] && echo --crop_borders || echo --no_crop_borders) ${EVAL_ARGS}
fi
elif [ -z "$ANALYSE_ONLY" ]; then
python -u -m dino.vis_video_results \
  --dataset_name "2025" \
  --splits "${SPLITS[@]}" \
  --annotations_path "$Q5_SPLITS" \
  --frame_annotations_path "$Q5_SPLITS" \
  --videos_path "data/ensemble_results_paxos2025/cleaned_videos" \
  --output_dir "$Q5_OUT" \
  --model Clf classifier $(ckpts model.pth "${CLF_PGL[@]}") \
  --model Clf32 classifier_video $(ckpts model.pth "${CLF_PGL[@]}") \
  --model DINO transformer $(ckpts sa_model.pth "${DINO_PGL[@]}") \
  --model DINO_FT transformer $(ckpts sa_model.pth "${DINO_FT_PGL[@]}") \
  --model CNX convnext_transformer $(ckpts sa_model.pth "${CNX_PGL[@]}") \
  --model CNX_mix convnext_transformer $(ckpts sa_model.pth "${CNX_MIX_PGL[@]}") \
  --img_size 512 \
  --complex_augs \
  --no_crop_borders
fi

# Reports (only once all 5 prediction CSVs exist)
if [ "$(ls "$Q5_OUT"/predictions_split_*.csv 2>/dev/null | wc -l)" -eq "${N_CSV:-5}" ]; then
  Q5_OUT=$(realpath "$Q5_OUT")
  cd data/multi_eval_paxos2025_all
  python make_binary.py --input_dir "$Q5_OUT"
  python analyse_runs.py --input_dir "$Q5_OUT" --output_dir "$Q5_OUT/split_analysis_results"
  python analyse_runs.py --input_dir "$Q5_OUT/binary" --output_dir "$Q5_OUT/binary/split_analysis_results" ${TXT_DIR:+--txt_dir "$TXT_DIR"}
  python plot_roc.py --input_dir "$Q5_OUT/binary"
fi
